#!/usr/bin/env python3
"""Compute a CDT-like CLIMDEX precipitation subset from daily NetCDF folder.

Implemented indices:
- Rx1day
- Rx5day
- R10mm
- R20mm
- Rnnmm
- CDD
- CWD
- PRCPTOT

Outputs:
- CLIMDEX_PRECIP_data/DATA_NetCDF/<INDEX>/Yearly/<index>_YYYY.nc
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from scipy.stats import t as student_t


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
    p = argparse.ArgumentParser(description="Compute CLIMDEX precip indices from daily NetCDF")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--var", default=None)
    p.add_argument("--indices", default="Rx1day,Rx5day,R10mm,R20mm,Rnnmm,CDD,CWD,PRCPTOT")
    p.add_argument("--rnn-threshold", type=float, default=25.0)
    p.add_argument("--min-frac", type=float, default=0.95)
    p.add_argument("--trend-min-years", type=int, default=20)
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)
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
            raise RuntimeError("lon/lat not detected")

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


def rolling_sum_5(ts: np.ndarray) -> np.ndarray:
    # ts: [time, points]
    out = np.full_like(ts, np.nan, dtype=np.float64)
    if ts.shape[0] < 5:
        return out
    csum = np.nancumsum(np.where(np.isnan(ts), 0.0, ts), axis=0)
    out[4:, :] = csum[4:, :] - np.vstack([np.zeros((1, ts.shape[1])), csum[:-5, :]])
    return out


def run_length_max(cond: np.ndarray) -> np.ndarray:
    # cond: [time, points] boolean, False where missing or condition not met.
    t, n = cond.shape
    out = np.zeros(n, dtype=np.int32)
    cur = np.zeros(n, dtype=np.int32)
    for i in range(t):
        cur = np.where(cond[i, :], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def write_annual_nc(path: Path, meta: Meta, var_name: str, data: np.ndarray, units: str, long_name: str, is_int: bool) -> None:
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

        dtype = "i2" if is_int else "f4"
        fill = np.int16(-9999) if is_int else np.float32(-9999.0)
        vv = ds.createVariable(var_name, dtype, (meta.lon_name, meta.lat_name), fill_value=fill, zlib=True, complevel=9)

        out = np.array(data, dtype=np.float64)
        out[~np.isfinite(out)] = -9999
        vv[:] = out.astype(np.int16 if is_int else np.float32)
        vv.units = units
        vv.long_name = long_name


def regression_vector(x: np.ndarray, y: np.ndarray, min_len: int) -> np.ndarray:
    # Equivalent structure to CDT regression.Vector: 10 x npoints
    # [slope, std.slope, t-value.slope, p-value.slope,
    #  intercept, std.intercept, t-value.intercept, p-value.intercept, R2, sigma]
    n_time, n_pts = y.shape
    res = np.full((10, n_pts), np.nan, dtype=np.float64)

    for j in range(n_pts):
        yy = y[:, j]
        ok = np.isfinite(x) & np.isfinite(yy)
        n = int(np.sum(ok))
        if n < min_len or n < 3:
            continue

        xx = x[ok]
        yy = yy[ok]

        mx = np.mean(xx)
        my = np.mean(yy)
        vx = np.var(xx, ddof=1)
        vy = np.var(yy, ddof=1)
        if not np.isfinite(vx) or vx <= 0:
            continue

        cov = np.sum((xx - mx) * (yy - my)) / (n - 1)
        alpha = cov / vx
        beta = my - alpha * mx

        yhat = alpha * xx + beta
        sse = np.sum((yhat - yy) ** 2)
        mse = sse / (n - 2)
        if mse < 0:
            continue
        sigma = np.sqrt(mse)

        sxx = (n - 1) * vx
        if sxx <= 0:
            continue

        std_alpha = sigma / (np.sqrt(n - 1) * np.sqrt(vx))
        std_beta = sigma * np.sqrt((1.0 / n) + (mx * mx / sxx))

        den_a = np.sqrt(mse / sxx) if mse > 0 else np.nan
        den_b = np.sqrt(mse * ((1.0 / n) + (mx * mx / sxx))) if mse > 0 else np.nan
        t_alpha = alpha / den_a if np.isfinite(den_a) and den_a > 0 else np.nan
        t_beta = beta / den_b if np.isfinite(den_b) and den_b > 0 else np.nan

        p_alpha = 2.0 * student_t.sf(np.abs(t_alpha), df=n - 2) if np.isfinite(t_alpha) else np.nan
        p_beta = 2.0 * student_t.sf(np.abs(t_beta), df=n - 2) if np.isfinite(t_beta) else np.nan

        r2 = (cov * cov) / (vx * vy) if np.isfinite(vy) and vy > 0 else np.nan

        res[:, j] = np.array(
            [alpha, std_alpha, t_alpha, p_alpha, beta, std_beta, t_beta, p_beta, r2, sigma],
            dtype=np.float64,
        )

    return res


def write_trend_nc(path: Path, meta: Meta, trend_data: np.ndarray) -> None:
    trend_names = [
        "slope",
        "std.slope",
        "t.value.slope",
        "p.value.slope",
        "intercept",
        "std.intercept",
        "t.value.intercept",
        "p.value.intercept",
        "R2",
        "sigma",
    ]
    trend_long = [
        "Slope - Estimate",
        "Slope - Standard Error",
        "Slope t-value",
        "Slope p-value Pr(>t)",
        "Intercept - Estimate",
        "Intercept - Standard Error",
        "Intercept t-value",
        "Intercept p-value Pr(>t)",
        "Multiple R-squared",
        "Residual Standard Error",
    ]

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

        for i, nm in enumerate(trend_names):
            vv = ds.createVariable(nm, "f4", (meta.lon_name, meta.lat_name), fill_value=np.float32(-9999.0), zlib=True, complevel=9)
            arr = trend_data[i, :].reshape((len(meta.lon), len(meta.lat)))
            arr = np.array(arr, dtype=np.float64)
            arr[~np.isfinite(arr)] = -9999.0
            vv[:] = arr.astype(np.float32)
            vv.units = ""
            vv.long_name = trend_long[i]


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    inp = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    files = list_daily(inp)
    if not files:
        print("No daily NetCDF files found")
        return 1

    meta = detect_meta(files[0].path, args.var)
    for f in files:
        ensure_grid(f.path, meta)

    dates = [f.date for f in files]
    years = np.array([int(d[:4]) for d in dates], dtype=np.int32)

    y0 = int(np.min(years)) if args.year_start is None else args.year_start
    y1 = int(np.max(years)) if args.year_end is None else args.year_end
    use_years = [y for y in range(y0, y1 + 1)]

    arr = [read_field(f.path, meta).reshape(-1) for f in files]
    ts = np.stack(arr, axis=0)  # [time, points]

    chosen = {x.strip() for x in args.indices.split(",") if x.strip()}
    npt = ts.shape[1]

    defs: Dict[str, Tuple[str, str, bool]] = {
        "Rx1day": ("mm", "Monthly maximum 1-day precipitation", False),
        "Rx5day": ("mm", "Monthly maximum consecutive 5-day precipitation", False),
        "R10mm": ("days", "Annual count of days when PRCP >= 10mm", True),
        "R20mm": ("days", "Annual count of days when PRCP >= 20mm", True),
        "Rnnmm": ("days", "Annual count of days when PRCP >= nnmm", True),
        "CDD": ("days", "Maximum length of dry spell, RR < 1mm", True),
        "CWD": ("days", "Maximum length of wet spell, RR >= 1mm", True),
        "PRCPTOT": ("mm", "Annual total precipitation in wet days", False),
    }

    annual_by_index: Dict[str, np.ndarray] = {
        name: np.full((len(use_years), npt), np.nan, dtype=np.float64) for name in defs.keys()
    }
    use_years_arr = np.array(use_years, dtype=np.float64)

    rr5 = rolling_sum_5(ts)

    for y in use_years:
        iy_out = y - use_years[0]
        iy = np.where(years == y)[0]
        if iy.size == 0:
            continue

        sub = ts[iy, :]
        n_valid = np.sum(np.isfinite(sub), axis=0)
        enough = n_valid >= int(np.ceil(365 * args.min_frac * 0.9))

        out_vals: Dict[str, np.ndarray] = {}

        if "Rx1day" in chosen:
            v = np.nanmax(sub, axis=0)
            v[~enough] = np.nan
            out_vals["Rx1day"] = v

        if "Rx5day" in chosen:
            r5 = rr5[iy, :]
            v = np.nanmax(r5, axis=0)
            v[~enough] = np.nan
            out_vals["Rx5day"] = v

        if "R10mm" in chosen:
            v = np.sum((sub >= 10.0) & np.isfinite(sub), axis=0).astype(np.float64)
            v[~enough] = np.nan
            out_vals["R10mm"] = v

        if "R20mm" in chosen:
            v = np.sum((sub >= 20.0) & np.isfinite(sub), axis=0).astype(np.float64)
            v[~enough] = np.nan
            out_vals["R20mm"] = v

        if "Rnnmm" in chosen:
            v = np.sum((sub >= args.rnn_threshold) & np.isfinite(sub), axis=0).astype(np.float64)
            v[~enough] = np.nan
            out_vals["Rnnmm"] = v

        if "CDD" in chosen:
            cond = (sub < 1.0) & np.isfinite(sub)
            v = run_length_max(cond).astype(np.float64)
            v[~enough] = np.nan
            out_vals["CDD"] = v

        if "CWD" in chosen:
            cond = (sub >= 1.0) & np.isfinite(sub)
            v = run_length_max(cond).astype(np.float64)
            v[~enough] = np.nan
            out_vals["CWD"] = v

        if "PRCPTOT" in chosen:
            wet = np.where((sub >= 1.0) & np.isfinite(sub), sub, 0.0)
            v = np.nansum(wet, axis=0)
            v[~enough] = np.nan
            out_vals["PRCPTOT"] = v

        for name, vec in out_vals.items():
            annual_by_index[name][iy_out, :] = vec
            units, long_name, is_int = defs[name]
            grid = vec.reshape((len(meta.lon), len(meta.lat)))
            out_file = out_root / "CLIMDEX_PRECIP_data" / "DATA_NetCDF" / name / "Yearly" / f"{name.lower()}_{y}.nc"
            write_annual_nc(out_file, meta, name, grid, units, long_name, is_int)

        if args.verbose:
            print(f"Year {y} done")

    for name in chosen:
        if name not in annual_by_index:
            continue
        trend = regression_vector(use_years_arr, annual_by_index[name], args.trend_min_years)
        out_trend = out_root / "CLIMDEX_PRECIP_data" / "DATA_NetCDF" / name / "Trend" / f"{name}.nc"
        write_trend_nc(out_trend, meta, trend)

    print(f"Done. Output root: {out_root / 'CLIMDEX_PRECIP_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
