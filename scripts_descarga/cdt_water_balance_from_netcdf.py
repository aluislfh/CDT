#!/usr/bin/env python3
"""Compute daily water balance from precipitation and PET NetCDF folders.

CDT simple water balance form:
WB[t] = clip(WB[t-1] + P[t] - PET[t], 0, capacity_max)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import netCDF4 as nc
import numpy as np


LON_CANDIDATES = {"lon", "longitude", "x", "Lon", "Longitude"}
LAT_CANDIDATES = {"lat", "latitude", "y", "Lat", "Latitude"}


@dataclass
class GridFile:
    path: Path
    date: str  # YYYYMMDD


@dataclass
class Meta:
    var_name: str
    lon_name: str
    lat_name: str
    lon: np.ndarray
    lat: np.ndarray
    fill_value: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute daily water balance from precipitation and PET NetCDF")
    p.add_argument("--precip-dir", required=True)
    p.add_argument("--pet-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--precip-var", default=None)
    p.add_argument("--pet-var", default=None)
    p.add_argument("--capacity-max", type=float, default=100.0)
    p.add_argument("--initial-wb", type=float, default=0.0)
    p.add_argument("--capacity-grid", default=None, help="Optional NetCDF grid for per-pixel max capacity")
    p.add_argument("--initial-grid", default=None, help="Optional NetCDF grid for per-pixel initial WB")
    p.add_argument("--date-start", default=None)
    p.add_argument("--date-end", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def parse_yyyymmdd(path: Path) -> str | None:
    for txt in (path.stem, path.name):
        m = re.findall(r"(19\d{2}|20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", txt)
        if m:
            y, mo, dd = m[-1]
            return f"{y}{mo}{dd}"
    return None


def list_daily(folder: Path) -> List[GridFile]:
    out: List[GridFile] = []
    for p in sorted(folder.glob("*.nc")):
        d = parse_yyyymmdd(p)
        if d:
            out.append(GridFile(path=p, date=d))
    out.sort(key=lambda x: x.date)
    return out


def detect_meta(path: Path, requested_var: str | None) -> Meta:
    with nc.Dataset(path, "r") as ds:
        lon_name = None
        lat_name = None
        for n, v in ds.variables.items():
            if v.ndim != 1:
                continue
            if lon_name is None and n in LON_CANDIDATES:
                lon_name = n
            if lat_name is None and n in LAT_CANDIDATES:
                lat_name = n

        if lon_name is None:
            for c in ["lon", "longitude", "Lon", "Longitude"]:
                if c in ds.variables and ds.variables[c].ndim == 1:
                    lon_name = c
                    break
        if lat_name is None:
            for c in ["lat", "latitude", "Lat", "Latitude"]:
                if c in ds.variables and ds.variables[c].ndim == 1:
                    lat_name = c
                    break

        if lon_name is None or lat_name is None:
            raise RuntimeError("lon/lat not found")

        if requested_var:
            if requested_var not in ds.variables:
                raise RuntimeError(f"Variable not found: {requested_var}")
            var_name = requested_var
        else:
            var_name = None
            for n, v in ds.variables.items():
                if n in {lon_name, lat_name}:
                    continue
                if lon_name in v.dimensions and lat_name in v.dimensions:
                    var_name = n
                    break
            if var_name is None:
                raise RuntimeError("data variable not found")

        var = ds.variables[var_name]
        fill = float(getattr(var, "_FillValue", -99.0))
        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)

    return Meta(var_name=var_name, lon_name=lon_name, lat_name=lat_name, lon=lon, lat=lat, fill_value=fill)


def ensure_grid(path: Path, meta: Meta) -> None:
    with nc.Dataset(path, "r") as ds:
        lon = np.array(ds.variables[meta.lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[meta.lat_name][:], dtype=np.float64)
    if lon.shape != meta.lon.shape or lat.shape != meta.lat.shape:
        raise RuntimeError(f"grid mismatch: {path.name}")
    if not np.allclose(lon, meta.lon, equal_nan=True) or not np.allclose(lat, meta.lat, equal_nan=True):
        raise RuntimeError(f"coordinate mismatch: {path.name}")


def read_field(path: Path, meta: Meta) -> np.ndarray:
    with nc.Dataset(path, "r") as ds:
        var = ds.variables[meta.var_name]
        arr = np.array(var[:], dtype=np.float64)
        dims = list(var.dimensions)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"unsupported variable dims in {path.name}")

    ax_lon = dims.index(meta.lon_name) if meta.lon_name in dims else 0
    ax_lat = dims.index(meta.lat_name) if meta.lat_name in dims else 1
    if not (ax_lon == 0 and ax_lat == 1):
        arr = np.transpose(arr, (ax_lon, ax_lat))

    arr[np.isclose(arr, meta.fill_value, equal_nan=False)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def read_optional_grid(path: str, ref_meta: Meta) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"Grid file not found: {p}")
    meta = detect_meta(p, None)
    ensure_grid(p, ref_meta)
    return read_field(p, meta)


def write_wb_nc(path: Path, meta: Meta, wb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension(meta.lon_name, len(meta.lon))
        ds.createDimension(meta.lat_name, len(meta.lat))

        vlon = ds.createVariable(meta.lon_name, "f4", (meta.lon_name,))
        vlat = ds.createVariable(meta.lat_name, "f4", (meta.lat_name,))
        vlon[:] = meta.lon.astype(np.float32)
        vlat[:] = meta.lat.astype(np.float32)
        vlon.units = "degrees_east"
        vlat.units = "degrees_north"

        vv = ds.createVariable("wb", "f4", (meta.lon_name, meta.lat_name), fill_value=np.float32(-9999.0), zlib=True, complevel=9)
        out = np.array(wb, dtype=np.float64)
        out[~np.isfinite(out)] = -9999.0
        vv[:] = out.astype(np.float32)
        vv.units = "mm"
        vv.long_name = "Daily water balance"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    pdir = Path(args.precip_dir).expanduser().resolve()
    edir = Path(args.pet_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    pfiles = list_daily(pdir)
    efiles = list_daily(edir)
    if not pfiles or not efiles:
        print("No daily NetCDF files found")
        return 1

    pmap: Dict[str, GridFile] = {f.date: f for f in pfiles}
    emap: Dict[str, GridFile] = {f.date: f for f in efiles}
    dates = sorted(set(pmap.keys()) & set(emap.keys()))

    if args.date_start:
        dates = [d for d in dates if d >= args.date_start]
    if args.date_end:
        dates = [d for d in dates if d <= args.date_end]

    if not dates:
        print("No common dates")
        return 1

    pmeta = detect_meta(pmap[dates[0]].path, args.precip_var)
    emeta = detect_meta(emap[dates[0]].path, args.pet_var)

    for d in dates:
        ensure_grid(pmap[d].path, pmeta)
        ensure_grid(emap[d].path, emeta)

    if len(pmeta.lon) != len(emeta.lon) or len(pmeta.lat) != len(emeta.lat):
        print("Grid size mismatch between precip and PET")
        return 1
    if not np.allclose(pmeta.lon, emeta.lon, equal_nan=True) or not np.allclose(pmeta.lat, emeta.lat, equal_nan=True):
        print("Grid coordinates mismatch between precip and PET")
        return 1

    if args.capacity_grid:
        cap = read_optional_grid(args.capacity_grid, pmeta)
    else:
        cap = np.full((len(pmeta.lon), len(pmeta.lat)), args.capacity_max, dtype=np.float64)

    if args.initial_grid:
        wb_prev = read_optional_grid(args.initial_grid, pmeta)
    else:
        wb_prev = np.full((len(pmeta.lon), len(pmeta.lat)), args.initial_wb, dtype=np.float64)

    wb_prev = np.clip(wb_prev, 0.0, cap)

    out_dir = out_root / "WB_data" / "DATA_NetCDF" / "WaterBalance_daily"

    for d in dates:
        rr = read_field(pmap[d].path, pmeta)
        et = read_field(emap[d].path, emeta)

        rr0 = rr.copy()
        et0 = et.copy()
        rr0[np.isnan(rr0)] = 0.0
        et0[np.isnan(et0)] = 0.0

        wb = wb_prev + rr0 - et0
        wb = np.clip(wb, 0.0, cap)

        # Keep all-NA pixels as missing if both inputs are missing this day.
        both_na = np.isnan(rr) & np.isnan(et)
        wb[both_na] = np.nan

        write_wb_nc(out_dir / f"wb_{d}.nc", pmeta, wb)
        wb_prev = np.where(np.isnan(wb), wb_prev, wb)

    if args.verbose:
        print(f"Processed {len(dates)} daily steps")
    print(f"Done. Output root: {out_root / 'WB_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
