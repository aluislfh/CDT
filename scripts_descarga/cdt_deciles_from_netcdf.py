#!/usr/bin/env python3
"""Compute CDT-like Deciles from monthly/dekadal NetCDF folder.

Outputs are written under:
- DECILE_data/DATA_NetCDF/Decile_<scale>mon/decile_YYYYMM.nc
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import netCDF4 as nc
import numpy as np


LON_CANDIDATES = {"lon", "longitude", "x", "Lon", "Longitude"}
LAT_CANDIDATES = {"lat", "latitude", "y", "Lat", "Latitude"}


@dataclass
class GridFile:
    path: Path
    date: str


@dataclass
class Meta:
    var_name: str
    lon_name: str
    lat_name: str
    lon: np.ndarray
    lat: np.ndarray
    fill_value: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Deciles from NetCDF folder")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--var", default=None)
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--base-all-years", action="store_true")
    p.add_argument("--base-start-year", type=int, default=None)
    p.add_argument("--base-end-year", type=int, default=None)
    p.add_argument("--min-year", type=int, default=20)
    p.add_argument("--date-start", default=None)
    p.add_argument("--date-end", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def parse_yyyymm(path: Path) -> str | None:
    for txt in (path.stem, path.name):
        m = re.findall(r"(19\d{2}|20\d{2})(0[1-9]|1[0-2])", txt)
        if m:
            y, mo = m[-1]
            return f"{y}{mo}"
    return None


def list_nc(folder: Path) -> List[GridFile]:
    out: List[GridFile] = []
    for p in sorted(folder.glob("*.nc")):
        d = parse_yyyymm(p)
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

        if lon_name is None or lat_name is None:
            raise RuntimeError("lon/lat not detected")

        if requested_var:
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
                raise RuntimeError("data variable not detected")

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
        v = ds.variables[meta.var_name]
        arr = np.array(v[:], dtype=np.float64)
        dims = list(v.dimensions)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"unsupported dimensions: {path.name}")

    ax_lon = dims.index(meta.lon_name) if meta.lon_name in dims else 0
    ax_lat = dims.index(meta.lat_name) if meta.lat_name in dims else 1
    if not (ax_lon == 0 and ax_lat == 1):
        arr = np.transpose(arr, (ax_lon, ax_lat))

    arr[np.isclose(arr, meta.fill_value, equal_nan=False)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def write_decile_nc(path: Path, meta: Meta, data: np.ndarray) -> None:
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

        vv = ds.createVariable("decile", "i2", (meta.lon_name, meta.lat_name), fill_value=np.int16(-9999), zlib=True, complevel=9)
        out = np.array(data, dtype=np.float64)
        out[~np.isfinite(out)] = -9999
        vv[:] = out.astype(np.int16)
        vv.units = ""
        vv.long_name = "Deciles"


def aggregate_scale(mat: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return mat.copy()
    out = np.full_like(mat, np.nan, dtype=np.float64)
    for i in range(scale - 1, mat.shape[0]):
        out[i, :] = np.nansum(mat[i - scale + 1 : i + 1, :], axis=0)
    return out


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    inp = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    files = list_nc(inp)
    if not files:
        print("No NetCDF files found")
        return 1

    dates = [f.date for f in files]
    if args.date_start:
        files = [f for f in files if f.date >= args.date_start]
    if args.date_end:
        files = [f for f in files if f.date <= args.date_end]
    if not files:
        print("No files in selected date range")
        return 1

    dates = [f.date for f in files]

    meta = detect_meta(files[0].path, args.var)
    for f in files:
        ensure_grid(f.path, meta)

    arr = [read_field(f.path, meta).reshape(-1) for f in files]
    mat = np.stack(arr, axis=0)
    mat_aggr = aggregate_scale(mat, args.scale)

    years = np.array([int(d[:4]) for d in dates], dtype=np.int32)
    months = np.array([int(d[4:6]) for d in dates], dtype=np.int32)

    if args.base_all_years:
        base_sel = np.ones(len(dates), dtype=bool)
    else:
        if args.base_start_year is None or args.base_end_year is None:
            print("Specify --base-start-year and --base-end-year or use --base-all-years")
            return 2
        base_sel = (years >= args.base_start_year) & (years <= args.base_end_year)

    ntime, npt = mat_aggr.shape
    out = np.full((ntime, npt), np.nan, dtype=np.float64)

    probs = np.linspace(0.0, 1.0, 11)

    for mo in range(1, 13):
        idx_all = np.where(months == mo)[0]
        idx_base = idx_all[base_sel[idx_all]]

        if idx_base.size < args.min_year:
            continue

        sub = mat_aggr[idx_base, :]

        valid_count = np.sum(np.isfinite(sub), axis=0)
        ok_col = valid_count >= args.min_year
        if not np.any(ok_col):
            continue

        q = np.full((11, npt), np.nan, dtype=np.float64)
        q[:, ok_col] = np.nanquantile(sub[:, ok_col], probs, axis=0, method="hazen")

        target = mat_aggr[idx_all, :]
        dec = np.full(target.shape, np.nan, dtype=np.float64)

        for j in np.where(ok_col)[0]:
            x = target[:, j]
            qq = q[:, j]
            good = np.isfinite(x)
            if not np.any(good):
                continue
            dec[good, j] = np.searchsorted(qq, x[good], side="right")

        out[idx_all, :] = dec

    suffix = f"{args.scale}mon"
    out_dir = out_root / "DECILE_data" / "DATA_NetCDF" / f"Decile_{suffix}"

    for i, d in enumerate(dates):
        if i < (args.scale - 1):
            continue
        z = out[i, :].reshape((len(meta.lon), len(meta.lat)))
        write_decile_nc(out_dir / f"decile_{d}.nc", meta, z)

    if args.verbose:
        print(f"Processed {len(dates)} dates")
    print(f"Done. Output root: {out_root / 'DECILE_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
