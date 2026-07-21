#!/usr/bin/env python3
"""Compute CDT-like SPEI from monthly precipitation and PET NetCDF folders.

Outputs are written under:
- SPEI_data

This script follows CDT structure conceptually:
- aggregate by scale
- fit distribution by calendar month
- compute standardized index
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from scipy.stats import gamma as sp_gamma
from scipy.stats import norm


LON_CANDIDATES = {"lon", "longitude", "x", "Lon", "Longitude"}
LAT_CANDIDATES = {"lat", "latitude", "y", "Lat", "Latitude"}


@dataclass
class GridFile:
    path: Path
    date: str  # YYYYMM


@dataclass
class Meta:
    var_name: str
    lon_name: str
    lat_name: str
    lon: np.ndarray
    lat: np.ndarray
    fill_value: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SPEI from precip and PET monthly NetCDF folders.")
    p.add_argument("--precip-dir", required=True)
    p.add_argument("--pet-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--precip-var", default=None)
    p.add_argument("--pet-var", default=None)
    p.add_argument("--scales", default="1,3,6,12,24")
    p.add_argument("--distribution", choices=["gamma", "zscore"], default="gamma")
    p.add_argument("--date-start", default=None, help="YYYYMM")
    p.add_argument("--date-end", default=None, help="YYYYMM")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def parse_yyyymm(path: Path) -> str | None:
    for txt in (path.stem, path.name):
        m = re.findall(r"(19\d{2}|20\d{2})(0[1-9]|1[0-2])", txt)
        if m:
            y, mo = m[-1]
            return f"{y}{mo}"
    return None


def list_monthly_nc(folder: Path) -> List[GridFile]:
    out: List[GridFile] = []
    for p in sorted(folder.glob("*.nc")):
        d = parse_yyyymm(p)
        if d:
            out.append(GridFile(path=p, date=d))
    out.sort(key=lambda x: x.date)
    return out


def detect_meta(path: Path, requested_var: str | None) -> Meta:
    with nc.Dataset(path, mode="r") as ds:
        lon_name = None
        lat_name = None
        for name, var in ds.variables.items():
            if var.ndim != 1:
                continue
            if lon_name is None and name in LON_CANDIDATES:
                lon_name = name
            if lat_name is None and name in LAT_CANDIDATES:
                lat_name = name

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
            raise RuntimeError("Unable to detect lon/lat")

        if requested_var:
            if requested_var not in ds.variables:
                raise RuntimeError(f"Variable not found: {requested_var}")
            var_name = requested_var
        else:
            var_name = None
            for name, var in ds.variables.items():
                if name in {lon_name, lat_name}:
                    continue
                if lon_name in var.dimensions and lat_name in var.dimensions:
                    var_name = name
                    break
            if var_name is None:
                raise RuntimeError("Unable to auto-detect variable")

        var = ds.variables[var_name]
        fill = float(getattr(var, "_FillValue", -99.0))
        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)

    return Meta(var_name=var_name, lon_name=lon_name, lat_name=lat_name, lon=lon, lat=lat, fill_value=fill)


def ensure_grid(path: Path, meta: Meta) -> None:
    with nc.Dataset(path, mode="r") as ds:
        lon = np.array(ds.variables[meta.lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[meta.lat_name][:], dtype=np.float64)
    if lon.shape != meta.lon.shape or lat.shape != meta.lat.shape:
        raise RuntimeError(f"Grid size mismatch: {path.name}")
    if not np.allclose(lon, meta.lon, equal_nan=True) or not np.allclose(lat, meta.lat, equal_nan=True):
        raise RuntimeError(f"Grid coordinate mismatch: {path.name}")


def read_field(path: Path, meta: Meta) -> np.ndarray:
    with nc.Dataset(path, mode="r") as ds:
        var = ds.variables[meta.var_name]
        arr = np.array(var[:], dtype=np.float64)
        dims = list(var.dimensions)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Unsupported dimensions in {path.name}: {dims}")

    ax_lon = dims.index(meta.lon_name) if meta.lon_name in dims else 0
    ax_lat = dims.index(meta.lat_name) if meta.lat_name in dims else 1
    if not (ax_lon == 0 and ax_lat == 1):
        arr = np.transpose(arr, (ax_lon, ax_lat))

    arr[np.isclose(arr, meta.fill_value, equal_nan=False)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def write_spi_nc(path: Path, meta: Meta, data: np.ndarray, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension(meta.lon_name, len(meta.lon))
        ds.createDimension(meta.lat_name, len(meta.lat))

        vlon = ds.createVariable(meta.lon_name, "f4", (meta.lon_name,))
        vlat = ds.createVariable(meta.lat_name, "f4", (meta.lat_name,))
        vlon[:] = meta.lon.astype(np.float32)
        vlat[:] = meta.lat.astype(np.float32)
        vlon.units = "degrees_east"
        vlat.units = "degrees_north"

        vv = ds.createVariable(name, "f4", (meta.lon_name, meta.lat_name), fill_value=np.float32(-9999.0), zlib=True, complevel=9)
        out = np.array(data, dtype=np.float64)
        out[~np.isfinite(out)] = -9999.0
        vv[:] = out.astype(np.float32)
        vv.units = ""
        vv.long_name = "Standardized Precipitation Evapotranspiration Index"


def aggregate_scale(mat: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return mat.copy()
    out = np.full_like(mat, np.nan, dtype=np.float64)
    for i in range(scale - 1, mat.shape[0]):
        out[i, :] = np.nansum(mat[i - scale + 1 : i + 1, :], axis=0)
    return out


def fit_gamma(x: np.ndarray) -> Tuple[float, float, float] | None:
    # shape, scale, pzero
    v = x[np.isfinite(x)]
    if v.size < 5:
        return None
    pzero = float(np.mean(v == 0))
    vp = v[v > 0]
    if vp.size < 5:
        return None

    if np.unique(vp).size == 1:
        vp = vp + np.random.default_rng(1234).uniform(0.1, 0.5, size=vp.size)

    try:
        a, _, b = sp_gamma.fit(vp, floc=0)
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            return None
    except Exception:
        return None

    return float(a), float(b), pzero


def compute_gamma_index(agg: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.full_like(agg, np.nan, dtype=np.float64)
    npt = agg.shape[1]

    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        if idx.size == 0:
            continue
        sub = agg[idx, :]

        for j in range(npt):
            pars = fit_gamma(sub[:, j])
            if pars is None:
                continue
            a, b, pzero = pars
            x = sub[:, j]
            good = np.isfinite(x)
            if np.sum(good) < 4:
                continue

            cdf = np.full(x.shape, np.nan, dtype=np.float64)
            cdf[good] = sp_gamma.cdf(np.maximum(x[good], 0.0), a=a, loc=0, scale=b)
            cdf[good] = pzero + (1.0 - pzero) * cdf[good]
            cdf[good] = np.clip(cdf[good], 1e-8, 1 - 1e-8)
            z = norm.ppf(cdf)
            z[np.isneginf(z)] = -5
            z[np.isposinf(z)] = 5
            out[idx, j] = z

    return out


def compute_zscore_index(agg: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.full_like(agg, np.nan, dtype=np.float64)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        if idx.size == 0:
            continue
        sub = agg[idx, :]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=0)
        z = (sub - mu) / sd
        z[~np.isfinite(z)] = 0.0
        out[idx, :] = z
    return out


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    pdir = Path(args.precip_dir).expanduser().resolve()
    edir = Path(args.pet_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    if not pdir.exists() or not edir.exists():
        print("precip-dir or pet-dir not found")
        return 2

    pfiles = list_monthly_nc(pdir)
    efiles = list_monthly_nc(edir)
    if not pfiles or not efiles:
        print("No monthly NetCDF files found")
        return 1

    pmap = {x.date: x for x in pfiles}
    emap = {x.date: x for x in efiles}
    dates = sorted(set(pmap.keys()) & set(emap.keys()))

    if args.date_start:
        dates = [d for d in dates if d >= args.date_start]
    if args.date_end:
        dates = [d for d in dates if d <= args.date_end]

    if not dates:
        print("No common dates between precip and pet in selected range")
        return 1

    try:
        pmeta = detect_meta(pmap[dates[0]].path, args.precip_var)
        emeta = detect_meta(emap[dates[0]].path, args.pet_var)

        for d in dates:
            ensure_grid(pmap[d].path, pmeta)
            ensure_grid(emap[d].path, emeta)

        if len(pmeta.lon) != len(emeta.lon) or len(pmeta.lat) != len(emeta.lat):
            raise RuntimeError("Precip and PET grids differ")
        if not np.allclose(pmeta.lon, emeta.lon, equal_nan=True) or not np.allclose(pmeta.lat, emeta.lat, equal_nan=True):
            raise RuntimeError("Precip and PET coordinate values differ")
    except Exception as exc:
        print(f"Input validation failed: {exc}")
        return 1

    wb = []
    for d in dates:
        p = read_field(pmap[d].path, pmeta)
        e = read_field(emap[d].path, emeta)
        wb.append((p - e).reshape(-1))
    mat = np.stack(wb, axis=0)

    months = np.array([int(d[4:6]) for d in dates], dtype=np.int32)
    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]

    spi_root = out_root / "SPEI_data" / "DATA_NetCDF"

    for scale in scales:
        agg = aggregate_scale(mat, scale)
        if args.distribution == "gamma":
            idx_vals = compute_gamma_index(agg, months)
        else:
            idx_vals = compute_zscore_index(agg, months)

        subdir = spi_root / f"SPEI_{scale}mon"
        for i, d in enumerate(dates):
            if i < (scale - 1):
                continue
            z = idx_vals[i, :].reshape((len(pmeta.lon), len(pmeta.lat)))
            out_file = subdir / f"spei_{d}.nc"
            write_spi_nc(out_file, pmeta, z, "spei")

        if args.verbose:
            print(f"SPEI scale {scale} written to {subdir}")

    print(f"Done. Output root: {out_root / 'SPEI_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
