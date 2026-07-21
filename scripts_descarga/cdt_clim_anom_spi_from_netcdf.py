#!/usr/bin/env python3
"""Compute CDT-like climatology, anomalies and SPI from monthly NetCDF folders.

This script follows the workflow implemented in CDT R code for NetCDF datasets:
- CLIMATOLOGY_data
- ANOMALIES_data
- SPI_data

It is intended for monthly precipitation-like grids stored as one NetCDF per time step.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from scipy.stats import gamma as sp_gamma
from scipy.stats import norm


LON_CANDIDATES = {"lon", "longitude", "x", "Lon", "Longitude"}
LAT_CANDIDATES = {"lat", "latitude", "y", "Lat", "Latitude"}
FILL_DEFAULT = -99.0


@dataclass
class GridData:
    path: Path
    date: str  # YYYYMM


@dataclass
class FieldMeta:
    var_name: str
    lon_name: str
    lat_name: str
    units: str
    long_name: str
    fill_value: float
    lon: np.ndarray
    lat: np.ndarray


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute CDT-like climatology/anomaly/SPI products from monthly NetCDF files."
    )
    p.add_argument("--input-dir", required=True, help="Folder with monthly NetCDF files.")
    p.add_argument("--output-dir", required=True, help="Root output folder where products are created.")
    p.add_argument(
        "--operation",
        choices=["climatology", "anomaly", "spi", "all"],
        default="all",
        help="Which product to compute.",
    )
    p.add_argument(
        "--var",
        default=None,
        help="Variable name in source NetCDF. If omitted, auto-detect first non-coordinate variable.",
    )
    p.add_argument(
        "--all-years",
        action="store_true",
        help="Use all available years for climatology (default behavior).",
    )
    p.add_argument("--start-year", type=int, default=None, help="Climatology start year if not all-years.")
    p.add_argument("--end-year", type=int, default=None, help="Climatology end year if not all-years.")
    p.add_argument("--min-years", type=int, default=15, help="Minimum years required per month/pixel.")
    p.add_argument(
        "--anomaly-type",
        choices=["Difference", "Percentage", "Standardized"],
        default="Difference",
        help="Anomaly type as in CDT.",
    )
    p.add_argument(
        "--climatology-dir",
        default=None,
        help="Optional existing CLIMATOLOGY_data dir used by anomaly operation.",
    )
    p.add_argument(
        "--spi-scales",
        default="1,3,6,12,24",
        help="Comma-separated SPI scales in months.",
    )
    p.add_argument(
        "--spi-distribution",
        choices=["gamma", "zscore"],
        default="gamma",
        help="SPI distribution.",
    )
    p.add_argument(
        "--date-start",
        default=None,
        help="Optional output start date YYYYMM for anomaly/SPI.",
    )
    p.add_argument(
        "--date-end",
        default=None,
        help="Optional output end date YYYYMM for anomaly/SPI.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def parse_month_from_name(path: Path) -> str | None:
    stems = [path.stem, path.name]
    for s in stems:
        matches = re.findall(r"(19\d{2}|20\d{2})(0[1-9]|1[0-2])", s)
        if matches:
            y, m = matches[-1]
            return f"{y}{m}"
    return None


def list_input_files(input_dir: Path) -> List[GridData]:
    out = []
    for p in sorted(input_dir.glob("*.nc")):
        d = parse_month_from_name(p)
        if d is not None:
            out.append(GridData(path=p, date=d))
    out.sort(key=lambda x: x.date)
    return out


def detect_coords_and_var(ds: nc.Dataset, requested_var: str | None) -> Tuple[str, str, str]:
    lon_name = None
    lat_name = None

    for name, var in ds.variables.items():
        if var.ndim != 1:
            continue
        if name in LON_CANDIDATES and lon_name is None:
            lon_name = name
        if name in LAT_CANDIDATES and lat_name is None:
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
        raise RuntimeError("Unable to detect lon/lat coordinates")

    if requested_var:
        if requested_var not in ds.variables:
            raise RuntimeError(f"Variable {requested_var} not found in NetCDF")
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
            raise RuntimeError("Unable to auto-detect data variable")

    return lon_name, lat_name, var_name


def load_field_meta(sample_file: Path, requested_var: str | None) -> FieldMeta:
    with nc.Dataset(sample_file, mode="r") as ds:
        lon_name, lat_name, var_name = detect_coords_and_var(ds, requested_var)
        var = ds.variables[var_name]

        units = str(getattr(var, "units", ""))
        long_name = str(getattr(var, "long_name", var_name))

        if hasattr(var, "_FillValue"):
            fill_value = float(var._FillValue)
        elif hasattr(var, "missing_value"):
            mv = var.missing_value
            fill_value = float(mv[0] if isinstance(mv, np.ndarray) else mv)
        else:
            fill_value = FILL_DEFAULT

        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)

    return FieldMeta(
        var_name=var_name,
        lon_name=lon_name,
        lat_name=lat_name,
        units=units,
        long_name=long_name,
        fill_value=fill_value,
        lon=lon,
        lat=lat,
    )


def read_field(path: Path, meta: FieldMeta) -> np.ndarray:
    with nc.Dataset(path, mode="r") as ds:
        var = ds.variables[meta.var_name]
        arr = np.array(var[:], dtype=np.float64)
        dims = list(var.dimensions)

    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise RuntimeError(f"Unsupported variable dimensions in {path.name}: {dims}")

    lon_axis = dims.index(meta.lon_name) if meta.lon_name in dims else 0
    lat_axis = dims.index(meta.lat_name) if meta.lat_name in dims else 1
    if lon_axis == lat_axis:
        raise RuntimeError(f"Invalid lon/lat axes in {path.name}")

    if arr.ndim == 2 and not (lon_axis == 0 and lat_axis == 1):
        arr = np.transpose(arr, (lon_axis, lat_axis))

    arr[np.isclose(arr, meta.fill_value, equal_nan=False)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def ensure_same_grid(path: Path, meta: FieldMeta) -> None:
    with nc.Dataset(path, mode="r") as ds:
        lon = np.array(ds.variables[meta.lon_name][:], dtype=np.float64)
        lat = np.array(ds.variables[meta.lat_name][:], dtype=np.float64)
    if lon.shape != meta.lon.shape or lat.shape != meta.lat.shape:
        raise RuntimeError(f"Grid size mismatch in {path.name}")
    if not np.allclose(lon, meta.lon, equal_nan=True) or not np.allclose(lat, meta.lat, equal_nan=True):
        raise RuntimeError(f"Grid coordinates mismatch in {path.name}")


def write_single_nc(out_file: Path, meta: FieldMeta, var_name: str, units: str, long_name: str, data: np.ndarray, fill_value: float) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_file, mode="w", format="NETCDF4") as ds:
        ds.createDimension(meta.lon_name, len(meta.lon))
        ds.createDimension(meta.lat_name, len(meta.lat))

        vlon = ds.createVariable(meta.lon_name, "f4", (meta.lon_name,))
        vlat = ds.createVariable(meta.lat_name, "f4", (meta.lat_name,))
        vlon[:] = meta.lon.astype(np.float32)
        vlat[:] = meta.lat.astype(np.float32)
        vlon.units = "degrees_east"
        vlon.long_name = "Longitude"
        vlat.units = "degrees_north"
        vlat.long_name = "Latitude"

        vv = ds.createVariable(
            var_name,
            "f4",
            (meta.lon_name, meta.lat_name),
            fill_value=np.float32(fill_value),
            zlib=True,
            complevel=9,
        )
        arr = np.array(data, dtype=np.float64)
        arr[np.isnan(arr)] = fill_value
        vv[:] = arr.astype(np.float32)
        vv.units = units
        vv.long_name = long_name


def monthly_index(date_yyyymm: str) -> int:
    return int(date_yyyymm[4:6])


def filter_dates(files: List[GridData], date_start: str | None, date_end: str | None) -> List[GridData]:
    out = files
    if date_start:
        out = [x for x in out if x.date >= date_start]
    if date_end:
        out = [x for x in out if x.date <= date_end]
    return out


def compute_climatology(files: List[GridData], meta: FieldMeta, out_root: Path, all_years: bool, start_year: int | None, end_year: int | None, min_years: int, verbose: bool) -> Tuple[Path, Path]:
    clim_dir = out_root / "CLIMATOLOGY_data"
    mean_nc_dir = clim_dir / "DATA_NetCDF" / "CDTMEAN"
    std_nc_dir = clim_dir / "DATA_NetCDF" / "CDTSTD"

    selected = files
    if not all_years:
        if start_year is None or end_year is None:
            raise RuntimeError("start-year and end-year are required when all-years is false")
        selected = [x for x in files if start_year <= int(x.date[:4]) <= end_year]

    if not selected:
        raise RuntimeError("No files available for climatology period")

    month_groups: Dict[int, List[GridData]] = {m: [] for m in range(1, 13)}
    for g in selected:
        month_groups[monthly_index(g.date)].append(g)

    for m in range(1, 13):
        grp = month_groups[m]
        if not grp:
            clim_mean = np.full((len(meta.lon), len(meta.lat)), np.nan, dtype=np.float64)
            clim_std = np.full((len(meta.lon), len(meta.lat)), np.nan, dtype=np.float64)
        else:
            cube = []
            for g in grp:
                ensure_same_grid(g.path, meta)
                cube.append(read_field(g.path, meta))
            cube = np.stack(cube, axis=0)

            valid_count = np.sum(np.isfinite(cube), axis=0)
            clim_mean = np.nanmean(cube, axis=0)
            clim_std = np.nanstd(cube, axis=0, ddof=0)

            clim_mean[valid_count < min_years] = np.nan
            clim_std[valid_count < min_years] = np.nan

        mid = f"{m:02d}"
        write_single_nc(
            mean_nc_dir / f"clim_{mid}.nc",
            meta,
            meta.var_name,
            meta.units,
            f"monthly climatology mean from: {meta.long_name}",
            clim_mean,
            FILL_DEFAULT,
        )
        write_single_nc(
            std_nc_dir / f"clim_{mid}.nc",
            meta,
            meta.var_name,
            meta.units,
            f"monthly climatology std from: {meta.long_name}",
            clim_std,
            FILL_DEFAULT,
        )

        if verbose:
            print(f"Climatology month {mid} written")

    return mean_nc_dir, std_nc_dir


def load_clim_maps(mean_dir: Path, std_dir: Path, meta: FieldMeta) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    means: Dict[int, np.ndarray] = {}
    stds: Dict[int, np.ndarray] = {}
    for m in range(1, 13):
        mid = f"{m:02d}"
        p_mean = mean_dir / f"clim_{mid}.nc"
        p_std = std_dir / f"clim_{mid}.nc"
        if not p_mean.exists() or not p_std.exists():
            raise RuntimeError(f"Missing climatology file for month {mid}")
        means[m] = read_field(p_mean, meta)
        stds[m] = read_field(p_std, meta)
    return means, stds


def compute_anomalies(files: List[GridData], meta: FieldMeta, out_root: Path, anomaly_type: str, clim_dir: Path, date_start: str | None, date_end: str | None, verbose: bool) -> None:
    mean_dir = clim_dir / "DATA_NetCDF" / "CDTMEAN"
    std_dir = clim_dir / "DATA_NetCDF" / "CDTSTD"
    means, stds = load_clim_maps(mean_dir, std_dir, meta)

    out_dir = out_root / "ANOMALIES_data" / "DATA_NetCDF" / "CDTANOM"

    targets = filter_dates(files, date_start, date_end)
    if not targets:
        raise RuntimeError("No files selected for anomaly computation")

    for g in targets:
        month = monthly_index(g.date)
        x = read_field(g.path, meta)
        mu = means[month]

        if anomaly_type == "Difference":
            anom = x - mu
            units = meta.units
        elif anomaly_type == "Percentage":
            anom = 100.0 * (x - mu) / (mu + 0.001)
            units = "percentage"
        else:
            sd = stds[month]
            anom = (x - mu) / sd
            units = ""

        anom[~np.isfinite(anom)] = np.nan

        out_file = out_dir / f"anomaly_{g.date}.nc"
        write_single_nc(out_file, meta, "anom", units, f"anomaly from: {meta.long_name}", anom, -9999.0)
        if verbose:
            print(f"Anomaly written: {out_file}")


def aggregate_scale(mat: np.ndarray, scale: int) -> np.ndarray:
    # mat shape: time x points
    if scale <= 1:
        return mat.copy()
    out = np.full_like(mat, np.nan, dtype=np.float64)
    for i in range(scale - 1, mat.shape[0]):
        # CDT uses colSums(..., na.rm=TRUE)
        out[i, :] = np.nansum(mat[i - scale + 1 : i + 1, :], axis=0)
    return out


def fit_spi_params_gamma(x: np.ndarray) -> Tuple[float, float, float] | None:
    # Returns (shape, scale, pzero)
    v = x[np.isfinite(x)]
    if v.size < 5:
        return None

    pzero = np.mean(v == 0)
    vp = v[v > 0]
    if vp.size < 5:
        return None

    if np.unique(vp).size == 1:
        vp = vp + np.random.default_rng(1234).uniform(0.1, 0.5, size=vp.size)

    try:
        a, loc, b = sp_gamma.fit(vp, floc=0)
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            return None
    except Exception:
        return None

    return float(a), float(b), float(pzero)


def spi_compute_gamma(agg: np.ndarray, months: np.ndarray) -> np.ndarray:
    # agg shape: time x points
    out = np.full_like(agg, np.nan, dtype=np.float64)
    npts = agg.shape[1]

    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if idx.size == 0:
            continue

        x = agg[idx, :]
        for j in range(npts):
            pars = fit_spi_params_gamma(x[:, j])
            if pars is None:
                continue

            a, b, pzero = pars
            xx = x[:, j]
            good = np.isfinite(xx)
            if np.sum(good) < 4:
                continue

            cdf = np.full(xx.shape, np.nan, dtype=np.float64)
            cdf[good] = sp_gamma.cdf(np.maximum(xx[good], 0.0), a=a, loc=0, scale=b)
            cdf[good] = pzero + (1.0 - pzero) * cdf[good]
            cdf[good] = np.clip(cdf[good], 1e-8, 1 - 1e-8)
            spi = norm.ppf(cdf)
            spi[np.isneginf(spi)] = -5
            spi[np.isposinf(spi)] = 5

            out[idx, j] = spi

    return out


def spi_compute_zscore(agg: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.full_like(agg, np.nan, dtype=np.float64)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if idx.size == 0:
            continue
        x = agg[idx, :]
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0, ddof=0)
        z = (x - mu) / sd
        z[~np.isfinite(z)] = 0.0
        out[idx, :] = z
    return out


def compute_spi(files: List[GridData], meta: FieldMeta, out_root: Path, scales: List[int], distribution: str, date_start: str | None, date_end: str | None, verbose: bool) -> None:
    targets = filter_dates(files, date_start, date_end)
    if not targets:
        raise RuntimeError("No files selected for SPI computation")

    # Keep sorted by time.
    targets = sorted(targets, key=lambda x: x.date)

    cube = []
    for g in targets:
        arr = read_field(g.path, meta)
        cube.append(arr.reshape(-1))
    mat = np.stack(cube, axis=0)  # time x points

    months = np.array([monthly_index(g.date) for g in targets], dtype=np.int32)

    spi_root = out_root / "SPI_data" / "DATA_NetCDF"

    for scale in scales:
        agg = aggregate_scale(mat, scale)
        if distribution == "gamma":
            spi_vals = spi_compute_gamma(agg, months)
        else:
            spi_vals = spi_compute_zscore(agg, months)

        subdir = spi_root / f"SPI_{scale}mon"
        for i, g in enumerate(targets):
            if i < (scale - 1):
                continue
            z = spi_vals[i, :].reshape((len(meta.lon), len(meta.lat)))
            z[~np.isfinite(z)] = np.nan
            out_file = subdir / f"spi_{g.date}.nc"
            write_single_nc(out_file, meta, "spi", "", "Standardized Precipitation Index", z, -9999.0)

        if verbose:
            print(f"SPI scale {scale} written in {subdir}")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 2

    files = list_input_files(in_dir)
    if not files:
        print("No monthly .nc files found (YYYYMM in name)")
        return 1

    try:
        meta = load_field_meta(files[0].path, args.var)
        for g in files[1:]:
            ensure_same_grid(g.path, meta)
    except Exception as exc:
        print(f"Input validation failed: {exc}")
        return 1

    do_clim = args.operation in {"climatology", "all", "anomaly"}
    do_anom = args.operation in {"anomaly", "all"}
    do_spi = args.operation in {"spi", "all"}

    try:
        clim_dir = Path(args.climatology_dir).expanduser().resolve() if args.climatology_dir else (out_dir / "CLIMATOLOGY_data")

        if do_clim and (not args.climatology_dir or args.operation in {"climatology", "all"}):
            all_years = True if args.all_years else (args.start_year is None and args.end_year is None)
            compute_climatology(
                files=files,
                meta=meta,
                out_root=out_dir,
                all_years=all_years,
                start_year=args.start_year,
                end_year=args.end_year,
                min_years=args.min_years,
                verbose=args.verbose,
            )

        if do_anom:
            compute_anomalies(
                files=files,
                meta=meta,
                out_root=out_dir,
                anomaly_type=args.anomaly_type,
                clim_dir=clim_dir,
                date_start=args.date_start,
                date_end=args.date_end,
                verbose=args.verbose,
            )

        if do_spi:
            scales = [int(x.strip()) for x in args.spi_scales.split(",") if x.strip()]
            compute_spi(
                files=files,
                meta=meta,
                out_root=out_dir,
                scales=scales,
                distribution=args.spi_distribution,
                date_start=args.date_start,
                date_end=args.date_end,
                verbose=args.verbose,
            )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"Done. Output root: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__('sys').argv[1:]))
