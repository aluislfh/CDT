#!/usr/bin/env python3
"""Compute a CDT-like CLIMDEX temperature subset from daily NetCDF folders.

Implemented indices:
- TXn, TXx, TNn, TNx
- SU, ID, FD, TR
- TX10p, TX90p, TN10p, TN90p
- WSDI, CSDI
- DTR, GSL

Outputs:
- CLIMDEX_TEMP_data/DATA_NetCDF/<INDEX>/Yearly/<index>_YYYY.nc
- CLIMDEX_TEMP_data/DATA_NetCDF/<INDEX>/Trend/<index>.nc
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
    p = argparse.ArgumentParser(description="Compute CLIMDEX temperature indices from daily NetCDF")
    p.add_argument("--tx-dir", required=True, help="Daily Tmax NetCDF folder")
    p.add_argument("--tn-dir", required=True, help="Daily Tmin NetCDF folder")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tx-var", default=None)
    p.add_argument("--tn-var", default=None)
    p.add_argument(
        "--indices",
        default="TXn,TXx,TNn,TNx,SU,ID,FD,TR,TX10p,TX90p,TN10p,TN90p,WSDI,CSDI,DTR,GSL",
    )
    p.add_argument("--min-frac", type=float, default=0.95)
    p.add_argument("--trend-min-years", type=int, default=20)
    p.add_argument("--base-all-years", action="store_true")
    p.add_argument("--base-start-year", type=int, default=None)
    p.add_argument("--base-end-year", type=int, default=None)
    p.add_argument("--base-min-year", type=int, default=20)
    p.add_argument("--base-window", type=int, default=5, help="Half-window in days for percentile climatology")
    p.add_argument("--upTX", type=float, default=25.0)
    p.add_argument("--loTX", type=float, default=0.0)
    p.add_argument("--upTN", type=float, default=20.0)
    p.add_argument("--loTN", type=float, default=0.0)
    p.add_argument("--thresGSL", type=float, default=5.0)
    p.add_argument("--dayGSL", type=int, default=6)
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


def run_length_max(cond: np.ndarray) -> np.ndarray:
    t, n = cond.shape
    out = np.zeros(n, dtype=np.int32)
    cur = np.zeros(n, dtype=np.int32)
    for i in range(t):
        cur = np.where(cond[i, :], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def run_length_total_days(cond: np.ndarray, min_run: int) -> np.ndarray:
    t, n = cond.shape
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        c = cond[:, j]
        k = 0
        total = 0
        while k < t:
            if c[k]:
                s = k
                while k < t and c[k]:
                    k += 1
                length = k - s
                if length >= min_run:
                    total += length
            else:
                k += 1
        out[j] = total
    return out


def regression_vector(x: np.ndarray, y: np.ndarray, min_len: int) -> np.ndarray:
    res = np.full((10, y.shape[1]), np.nan, dtype=np.float64)
    for j in range(y.shape[1]):
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
        fill = np.int16(-99) if is_int else np.float32(-99.0)
        vv = ds.createVariable(var_name, dtype, (meta.lon_name, meta.lat_name), fill_value=fill, zlib=True, complevel=9)

        out = np.array(data, dtype=np.float64)
        out[~np.isfinite(out)] = -99.0
        vv[:] = out.astype(np.int16 if is_int else np.float32)
        vv.units = units
        vv.long_name = long_name


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
            vv = ds.createVariable(nm, "f4", (meta.lon_name, meta.lat_name), fill_value=np.float32(-99.0), zlib=True, complevel=9)
            arr = trend_data[i, :].reshape((len(meta.lon), len(meta.lat)))
            arr = np.array(arr, dtype=np.float64)
            arr[~np.isfinite(arr)] = -99.0
            vv[:] = arr.astype(np.float32)
            vv.units = ""
            vv.long_name = trend_long[i]


def ymd_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def doy_366(d: date) -> int:
    return int(d.strftime("%j"))


def circular_doy_dist(a: np.ndarray, b: int) -> np.ndarray:
    # day-of-year distance in 366-day circular calendar
    da = np.abs(a - b)
    return np.minimum(da, 366 - da)


def compute_daily_percentiles(
    data: np.ndarray,
    dates: List[date],
    base_sel: np.ndarray,
    window: int,
    q: float,
    min_year: int,
) -> np.ndarray:
    ntime, npt = data.shape
    out = np.full((366, npt), np.nan, dtype=np.float64)

    doys = np.array([doy_366(d) for d in dates], dtype=np.int32)
    base_idx = np.where(base_sel)[0]

    for d in range(1, 367):
        in_win = circular_doy_dist(doys[base_idx], d) <= window
        idx = base_idx[in_win]
        if idx.size == 0:
            continue

        sub = data[idx, :]
        valid = np.sum(np.isfinite(sub), axis=0)
        ok = valid >= min_year
        if not np.any(ok):
            continue

        out[d - 1, ok] = np.nanquantile(sub[:, ok], q, axis=0, method="linear")

    return out


def gsl_one_year(tmean_year: np.ndarray, thres: float, days: int, min_frac: float) -> np.ndarray:
    # tmean_year: [time, points] for one full year timeline (NA for missing dates)
    nday, npt = tmean_year.shape
    out = np.full(npt, np.nan, dtype=np.float64)

    valid_frac = np.sum(np.isfinite(tmean_year), axis=0) / float(nday)
    ok = valid_frac >= min_frac
    if not np.any(ok):
        return out

    t = tmean_year[:, ok]
    warm = np.isfinite(t) & (t > thres)
    cold = np.isfinite(t) & (~warm)

    for j in range(t.shape[1]):
        # first warm run >= days
        s = None
        k = 0
        while k < nday:
            if warm[k, j]:
                i0 = k
                while k < nday and warm[k, j]:
                    k += 1
                if (k - i0) >= days:
                    s = i0
                    break
            else:
                k += 1
        if s is None:
            continue

        # first cold run >= days after s
        e = nday
        k = s + 1
        while k < nday:
            if cold[k, j]:
                i0 = k
                while k < nday and cold[k, j]:
                    k += 1
                if (k - i0) >= days:
                    e = i0
                    break
            else:
                k += 1

        out[np.where(ok)[0][j]] = float(e - s)

    return out


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    tx_dir = Path(args.tx_dir).expanduser().resolve()
    tn_dir = Path(args.tn_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    tx_files = list_daily(tx_dir)
    tn_files = list_daily(tn_dir)
    if not tx_files or not tn_files:
        print("No daily NetCDF files found in tx-dir or tn-dir")
        return 1

    tx_map = {f.date: f for f in tx_files}
    tn_map = {f.date: f for f in tn_files}
    dates_key = sorted(set(tx_map.keys()) & set(tn_map.keys()))
    if not dates_key:
        print("No common dates between TX and TN")
        return 1

    tx_meta = detect_meta(tx_map[dates_key[0]].path, args.tx_var)
    tn_meta = detect_meta(tn_map[dates_key[0]].path, args.tn_var)

    for d in dates_key:
        ensure_grid(tx_map[d].path, tx_meta)
        ensure_grid(tn_map[d].path, tn_meta)

    if len(tx_meta.lon) != len(tn_meta.lon) or len(tx_meta.lat) != len(tn_meta.lat):
        print("Grid size mismatch between TX and TN")
        return 1
    if not np.allclose(tx_meta.lon, tn_meta.lon, equal_nan=True) or not np.allclose(tx_meta.lat, tn_meta.lat, equal_nan=True):
        print("Grid coordinate mismatch between TX and TN")
        return 1

    tx_arr = [read_field(tx_map[d].path, tx_meta).reshape(-1) for d in dates_key]
    tn_arr = [read_field(tn_map[d].path, tn_meta).reshape(-1) for d in dates_key]

    tx_ts = np.stack(tx_arr, axis=0)
    tn_ts = np.stack(tn_arr, axis=0)

    dates = [ymd_to_date(d) for d in dates_key]
    years = np.array([d.year for d in dates], dtype=np.int32)

    y0 = int(np.min(years)) if args.year_start is None else args.year_start
    y1 = int(np.max(years)) if args.year_end is None else args.year_end
    use_years = [y for y in range(y0, y1 + 1)]
    years_arr = np.array(use_years, dtype=np.float64)

    if args.base_all_years:
        base_sel = np.ones(len(dates), dtype=bool)
    else:
        if args.base_start_year is None or args.base_end_year is None:
            print("Specify base period (--base-start-year/--base-end-year) or use --base-all-years")
            return 2
        base_sel = (years >= args.base_start_year) & (years <= args.base_end_year)

    npt = tx_ts.shape[1]
    chosen = {x.strip() for x in args.indices.split(",") if x.strip()}

    defs: Dict[str, Tuple[str, str, bool]] = {
        "TXn": ("degC", "Monthly minimum value of daily maximum temperature", False),
        "TXx": ("degC", "Monthly maximum value of daily maximum temperature", False),
        "TNn": ("degC", "Monthly minimum value of daily minimum temperature", False),
        "TNx": ("degC", "Monthly maximum value of daily minimum temperature", False),
        "SU": ("days", "Number of summer days", True),
        "ID": ("days", "Number of icing days", True),
        "FD": ("days", "Number of frost days", True),
        "TR": ("days", "Number of tropical nights", True),
        "TX10p": ("%", "Percentage of days when TX < 10th percentile", False),
        "TX90p": ("%", "Percentage of days when TX > 90th percentile", False),
        "TN10p": ("%", "Percentage of days when TN < 10th percentile", False),
        "TN90p": ("%", "Percentage of days when TN > 90th percentile", False),
        "WSDI": ("days", "Warm spell duration index", True),
        "CSDI": ("days", "Cold spell duration index", True),
        "DTR": ("degC", "Daily temperature range", False),
        "GSL": ("days", "Growing season length", True),
    }

    need_pct = any(k in chosen for k in ["TX10p", "TX90p", "TN10p", "TN90p", "WSDI", "CSDI"])
    if need_pct:
        q10_tx = compute_daily_percentiles(tx_ts, dates, base_sel, args.base_window, 0.1, args.base_min_year) - 1e-5
        q90_tx = compute_daily_percentiles(tx_ts, dates, base_sel, args.base_window, 0.9, args.base_min_year) + 1e-5
        q10_tn = compute_daily_percentiles(tn_ts, dates, base_sel, args.base_window, 0.1, args.base_min_year) - 1e-5
        q90_tn = compute_daily_percentiles(tn_ts, dates, base_sel, args.base_window, 0.9, args.base_min_year) + 1e-5
        doys = np.array([doy_366(d) for d in dates], dtype=np.int32)

    annual_by_index: Dict[str, np.ndarray] = {
        name: np.full((len(use_years), npt), np.nan, dtype=np.float64) for name in defs.keys()
    }

    for yi, y in enumerate(use_years):
        idx = np.where(years == y)[0]
        if idx.size == 0:
            continue

        tx = tx_ts[idx, :]
        tn = tn_ts[idx, :]

        n_valid_tx = np.sum(np.isfinite(tx), axis=0)
        n_valid_tn = np.sum(np.isfinite(tn), axis=0)
        enough_tx = n_valid_tx >= int(np.ceil(365 * args.min_frac * 0.9))
        enough_tn = n_valid_tn >= int(np.ceil(365 * args.min_frac * 0.9))

        out_vals: Dict[str, np.ndarray] = {}

        if "TXn" in chosen:
            v = np.nanmin(tx, axis=0)
            v[~enough_tx] = np.nan
            out_vals["TXn"] = v
        if "TXx" in chosen:
            v = np.nanmax(tx, axis=0)
            v[~enough_tx] = np.nan
            out_vals["TXx"] = v
        if "TNn" in chosen:
            v = np.nanmin(tn, axis=0)
            v[~enough_tn] = np.nan
            out_vals["TNn"] = v
        if "TNx" in chosen:
            v = np.nanmax(tn, axis=0)
            v[~enough_tn] = np.nan
            out_vals["TNx"] = v

        if "SU" in chosen:
            v = np.sum((tx > args.upTX) & np.isfinite(tx), axis=0).astype(np.float64)
            v[~enough_tx] = np.nan
            out_vals["SU"] = v
        if "ID" in chosen:
            v = np.sum((tx < args.loTX) & np.isfinite(tx), axis=0).astype(np.float64)
            v[~enough_tx] = np.nan
            out_vals["ID"] = v
        if "FD" in chosen:
            v = np.sum((tn < args.loTN) & np.isfinite(tn), axis=0).astype(np.float64)
            v[~enough_tn] = np.nan
            out_vals["FD"] = v
        if "TR" in chosen:
            v = np.sum((tn > args.upTN) & np.isfinite(tn), axis=0).astype(np.float64)
            v[~enough_tn] = np.nan
            out_vals["TR"] = v

        if "DTR" in chosen:
            dtr = tx - tn
            v = np.nanmean(dtr, axis=0)
            v[~(enough_tx & enough_tn)] = np.nan
            out_vals["DTR"] = v

        if need_pct:
            i_doy = doys[idx] - 1
            q10tx = q10_tx[i_doy, :]
            q90tx = q90_tx[i_doy, :]
            q10tn = q10_tn[i_doy, :]
            q90tn = q90_tn[i_doy, :]

            tx_m10 = tx - q10tx
            tx_m90 = tx - q90tx
            tn_m10 = tn - q10tn
            tn_m90 = tn - q90tn

            if "TX10p" in chosen:
                c = np.sum((tx_m10 < 0) & np.isfinite(tx_m10), axis=0)
                v = 100.0 * c / float(len(idx))
                v[~enough_tx] = np.nan
                out_vals["TX10p"] = v
            if "TX90p" in chosen:
                c = np.sum((tx_m90 > 0) & np.isfinite(tx_m90), axis=0)
                v = 100.0 * c / float(len(idx))
                v[~enough_tx] = np.nan
                out_vals["TX90p"] = v
            if "TN10p" in chosen:
                c = np.sum((tn_m10 < 0) & np.isfinite(tn_m10), axis=0)
                v = 100.0 * c / float(len(idx))
                v[~enough_tn] = np.nan
                out_vals["TN10p"] = v
            if "TN90p" in chosen:
                c = np.sum((tn_m90 > 0) & np.isfinite(tn_m90), axis=0)
                v = 100.0 * c / float(len(idx))
                v[~enough_tn] = np.nan
                out_vals["TN90p"] = v
            if "WSDI" in chosen:
                cond = (tx_m90 > 0) & np.isfinite(tx_m90)
                v = run_length_total_days(cond, 6)
                v[~enough_tx] = np.nan
                out_vals["WSDI"] = v
            if "CSDI" in chosen:
                cond = (tn_m10 < 0) & np.isfinite(tn_m10)
                v = run_length_total_days(cond, 6)
                v[~enough_tn] = np.nan
                out_vals["CSDI"] = v

        if "GSL" in chosen:
            # Build complete year timeline to match CDT behavior with missing dates filled.
            d0 = date(y, 1, 1)
            d1 = date(y, 12, 31)
            full_dates: List[date] = []
            dd = d0
            while dd <= d1:
                full_dates.append(dd)
                dd += timedelta(days=1)

            mapper = {dates[idx[k]]: k for k in range(len(idx))}
            tmean_full = np.full((len(full_dates), npt), np.nan, dtype=np.float64)
            tmean = (tx + tn) / 2.0
            for k, dcur in enumerate(full_dates):
                ik = mapper.get(dcur)
                if ik is not None:
                    tmean_full[k, :] = tmean[ik, :]

            v = gsl_one_year(tmean_full, args.thresGSL, args.dayGSL, args.min_frac)
            out_vals["GSL"] = v

        for name, vec in out_vals.items():
            annual_by_index[name][yi, :] = vec
            units, long_name, is_int = defs[name]
            grid = vec.reshape((len(tx_meta.lon), len(tx_meta.lat)))
            out_file = out_root / "CLIMDEX_TEMP_data" / "DATA_NetCDF" / name / "Yearly" / f"{name}_{y}.nc"
            write_annual_nc(out_file, tx_meta, name, grid, units, long_name, is_int)

        if args.verbose:
            print(f"Year {y} done")

    for name in chosen:
        if name not in annual_by_index:
            continue
        trend = regression_vector(years_arr, annual_by_index[name], args.trend_min_years)
        out_trend = out_root / "CLIMDEX_TEMP_data" / "DATA_NetCDF" / name / "Trend" / f"{name}.nc"
        write_trend_nc(out_trend, tx_meta, trend)

    print(f"Done. Output root: {out_root / 'CLIMDEX_TEMP_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
