#!/usr/bin/env python3
"""Download or read local GSMaP CLM daily/monthly and write CDT NetCDF.

This script mirrors core CDT logic from R/cdtDownloadRFE_gsmap.R for source
"gsmap.clm-gb" with temporal resolutions:
- daily
- monthly
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import netCDF4 as nc
import numpy as np

from defaults import DEFAULT_BBOX_HELP, DEFAULT_MAXLAT, DEFAULT_MAXLON, DEFAULT_MINLAT, DEFAULT_MINLON
from env_utils import load_dotenv


FTP_BASE = "ftp://hokusai.eorc.jaxa.jp"
FTP_DAILY_DIR = "climate/gnrt6/daily"
FTP_MONTHLY_DIR = "climate/gnrt6/monthly"

NX = 3600
NY = 1200
NVAL = NX * NY


@dataclass(frozen=True)
class ProductSpec:
    product: str
    tstep: str
    data_name: str
    out_name_fmt: str


PRODUCT_SPECS = {
    "daily": ProductSpec(
        product="daily",
        tstep="daily",
        data_name="GSMaP_CLM_daily",
        out_name_fmt="gsmap_clm_00z_{yyyymmdd}.nc",
    ),
    "monthly": ProductSpec(
        product="monthly",
        tstep="monthly",
        data_name="GSMaP_CLM_monthly",
        out_name_fmt="gsmap_clm_{yyyymm}.nc",
    ),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/read GSMaP CLM binary files and convert to CDT-compatible NetCDF."
        )
    )
    parser.add_argument(
        "--product",
        required=True,
        choices=sorted(PRODUCT_SPECS.keys()),
        help="Temporal product.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date: YYYYMMDD for daily, YYYYMM for monthly.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date: YYYYMMDD for daily, YYYYMM for monthly.",
    )
    parser.add_argument("--minlon", type=float, default=DEFAULT_MINLON, help=DEFAULT_BBOX_HELP)
    parser.add_argument("--maxlon", type=float, default=DEFAULT_MAXLON, help=DEFAULT_BBOX_HELP)
    parser.add_argument("--minlat", type=float, default=DEFAULT_MINLAT, help=DEFAULT_BBOX_HELP)
    parser.add_argument("--maxlat", type=float, default=DEFAULT_MAXLAT, help=DEFAULT_BBOX_HELP)
    parser.add_argument(
        "--source",
        choices=["ftp", "local"],
        default="ftp",
        help="Source mode: download from FTP or use local files.",
    )
    parser.add_argument(
        "--local-root",
        default="/home/adrian/wildfire-cathalac-platform/local/CDT/GSMaP",
        help="Root path containing local GSMaP files (used with --source local).",
    )
    parser.add_argument(
        "--ftp-user",
        default=os.environ.get("GSMAP_FTP_USER"),
        help="FTP user (defaults to GSMAP_FTP_USER from environment or .env).",
    )
    parser.add_argument(
        "--ftp-password",
        default=os.environ.get("GSMAP_FTP_PASSWORD"),
        help="FTP password (defaults to GSMAP_FTP_PASSWORD from environment or .env).",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Base output directory. Product folder is created inside it.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original .dat.gz files after conversion.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def iter_daily(start: str, end: str) -> Iterable[str]:
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    if d1 < d0:
        raise ValueError("end date must be >= start date")
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def iter_monthly(start: str, end: str) -> Iterable[str]:
    d0 = datetime.strptime(start, "%Y%m").date().replace(day=1)
    d1 = datetime.strptime(end, "%Y%m").date().replace(day=1)
    if d1 < d0:
        raise ValueError("end date must be >= start date")
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y%m")
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = date(year, month, 1)


def build_target_dates(spec: ProductSpec, start: str, end: str) -> List[str]:
    if spec.tstep == "daily":
        if len(start) != 8 or len(end) != 8:
            raise ValueError("for daily product, --start and --end must be YYYYMMDD")
        return list(iter_daily(start, end))

    if len(start) != 6 or len(end) != 6:
        raise ValueError("for monthly product, --start and --end must be YYYYMM")
    return list(iter_monthly(start, end))


def validate_bbox(minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    if not (-180.0 <= minlon <= 180.0 and -180.0 <= maxlon <= 180.0):
        raise ValueError("Longitude must be within [-180, 180].")
    if not (-90.0 <= minlat <= 90.0 and -90.0 <= maxlat <= 90.0):
        raise ValueError("Latitude must be within [-90, 90].")
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("Invalid bbox: min values must be <= max values.")


def remote_filename(spec: ProductSpec, dt: str) -> str:
    if spec.tstep == "daily":
        return f"gsmap_gnrt6.{dt}.0.1d.daily.00Z-23Z.dat.gz"
    return f"gsmap_gnrt6.{dt}.0.1d.monthly.dat.gz"


def output_filename(spec: ProductSpec, dt: str) -> str:
    if spec.tstep == "daily":
        return spec.out_name_fmt.format(yyyymmdd=dt)
    return spec.out_name_fmt.format(yyyymm=dt)


def ftp_url(spec: ProductSpec, dt: str) -> str:
    if spec.tstep == "daily":
        return f"{FTP_BASE}/{FTP_DAILY_DIR}/{dt[:4]}{dt[4:6]}/{remote_filename(spec, dt)}"
    return f"{FTP_BASE}/{FTP_MONTHLY_DIR}/{dt[:4]}/{remote_filename(spec, dt)}"


def local_source_path(local_root: Path, spec: ProductSpec, dt: str) -> Path:
    fname = remote_filename(spec, dt)

    # Most common layout observed by user: <root>/<YYYY>/<filename>
    p1 = local_root / dt[:4] / fname
    if p1.exists():
        return p1

    # Fallback: recursive search by exact filename.
    matches = list(local_root.rglob(fname))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Local file not found: {fname}")


def curl_download_ftp(url: str, dest: Path, user: str, password: str) -> bool:
    cmd = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--user",
        f"{user}:{password}",
        "--output",
        str(dest),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


def read_gsmap_binary(dat_gz_file: Path, tstep: str) -> np.ndarray:
    arr = np.fromfile(dat_gz_file, dtype="<f4")

    if tstep == "monthly":
        if arr.size < 2 * NVAL:
            raise ValueError(f"Invalid monthly data length in {dat_gz_file.name}")
        val1 = arr[:NVAL].astype(np.float64)
        val2 = arr[NVAL : 2 * NVAL].astype(np.float64)
        val1[val1 < 0] = np.nan
        val = val1 * val2
    else:
        if arr.size < NVAL:
            raise ValueError(f"Invalid daily data length in {dat_gz_file.name}")
        val = arr[:NVAL].astype(np.float64)
        val[val < 0] = np.nan

    return val


def convert_to_cdt_nc(src_dat_file: Path, out_nc_file: Path, spec: ProductSpec, minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    val = read_gsmap_binary(src_dat_file, spec.tstep)

    lon = np.arange(0.05, 0.05 + 0.1 * NX, 0.1, dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = np.arange(59.95, 59.95 - 0.1 * NY, -0.1, dtype=np.float64)

    ox = np.argsort(lon)
    oy = np.argsort(lat)
    lon = lon[ox]
    lat = lat[oy]

    val[~np.isfinite(val)] = np.nan
    # R matrix() fills by column; use Fortran order to match.
    val = np.reshape(val, (NX, NY), order="F")
    val = val[ox, :][:, oy]

    ix = (lon >= minlon) & (lon <= maxlon)
    iy = (lat >= minlat) & (lat <= maxlat)

    if not np.any(ix) or not np.any(iy):
        raise ValueError("bbox does not intersect GSMaP grid")

    lon = lon[ix]
    lat = lat[iy]
    val = val[np.ix_(ix, iy)]

    if spec.tstep == "daily":
        val = val * 24.0

    val = np.round(val, 2)
    val[np.isnan(val)] = -99.0

    with nc.Dataset(out_nc_file, mode="w", format="NETCDF4") as dst:
        dst.createDimension("lon", lon.size)
        dst.createDimension("lat", lat.size)

        lon_var = dst.createVariable("lon", "f4", ("lon",))
        lat_var = dst.createVariable("lat", "f4", ("lat",))
        lon_var[:] = lon.astype(np.float32)
        lat_var[:] = lat.astype(np.float32)
        lon_var.units = "degrees_east"
        lon_var.long_name = "Longitude"
        lat_var.units = "degrees_north"
        lat_var.long_name = "Latitude"

        precip = dst.createVariable(
            "precip",
            "f4",
            ("lon", "lat"),
            zlib=True,
            complevel=9,
            fill_value=np.float32(-99.0),
        )
        precip[:] = val.astype(np.float32)
        precip.units = "mm"
        precip.long_name = "Precipitation estimates"


def ensure_dat_from_gz(src_gz: Path, dat_file: Path) -> None:
    cmd = ["gzip", "-cd", str(src_gz)]
    with dat_file.open("wb") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to decompress {src_gz.name}")


def process_one(
    spec: ProductSpec,
    dt: str,
    product_dir: Path,
    orig_dir: Path,
    extr_dir: Path,
    source: str,
    local_root: Path,
    ftp_user: str,
    ftp_password: str,
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
    keep_original: bool,
    verbose: bool,
) -> bool:
    remote_name = remote_filename(spec, dt)
    out_name = output_filename(spec, dt)

    src_gz = orig_dir / remote_name
    out_nc = extr_dir / out_name

    if source == "ftp":
        url = ftp_url(spec, dt)
        if verbose:
            print(f"Downloading: {url}")
        ok = curl_download_ftp(url, src_gz, ftp_user, ftp_password)
        if not ok:
            print(f"FAILED: could not download {dt}", file=sys.stderr)
            return False
    else:
        try:
            local_file = local_source_path(local_root, spec, dt)
        except Exception as exc:
            print(f"FAILED: {dt}: {exc}", file=sys.stderr)
            return False

        if verbose:
            print(f"Using local file: {local_file}")
        shutil.copy2(local_file, src_gz)

    tmp_dat = extr_dir / remote_name.replace(".gz", "")

    try:
        ensure_dat_from_gz(src_gz, tmp_dat)
        convert_to_cdt_nc(tmp_dat, out_nc, spec, minlon, maxlon, minlat, maxlat)
        if verbose:
            print(f"Wrote: {out_nc}")
    except Exception as exc:
        print(f"FAILED: conversion error for {dt}: {exc}", file=sys.stderr)
        return False
    finally:
        if tmp_dat.exists():
            tmp_dat.unlink(missing_ok=True)
        if not keep_original and src_gz.exists():
            src_gz.unlink(missing_ok=True)

    return True


def main(argv: Sequence[str]) -> int:
    load_dotenv(Path(__file__).resolve().parent)
    args = parse_args(argv)
    spec = PRODUCT_SPECS[args.product]

    try:
        validate_bbox(args.minlon, args.maxlon, args.minlat, args.maxlat)
        dates = build_target_dates(spec, args.start, args.end)
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    if args.source == "ftp" and (not args.ftp_user or not args.ftp_password):
        print(
            "FTP credentials are required. Use --ftp-user/--ftp-password or "
            "GSMAP_FTP_USER/GSMAP_FTP_PASSWORD in environment or .env.",
            file=sys.stderr,
        )
        return 2

    product_dir = Path(args.outdir).expanduser().resolve() / spec.data_name
    orig_dir = product_dir / "Data_global"
    extr_dir = product_dir / "Extracted"
    orig_dir.mkdir(parents=True, exist_ok=True)
    extr_dir.mkdir(parents=True, exist_ok=True)

    local_root = Path(args.local_root).expanduser().resolve()

    failures = 0
    for dt in dates:
        ok = process_one(
            spec=spec,
            dt=dt,
            product_dir=product_dir,
            orig_dir=orig_dir,
            extr_dir=extr_dir,
            source=args.source,
            local_root=local_root,
            ftp_user=args.ftp_user,
            ftp_password=args.ftp_password,
            minlon=args.minlon,
            maxlon=args.maxlon,
            minlat=args.minlat,
            maxlat=args.maxlat,
            keep_original=args.keep_original,
            verbose=args.verbose,
        )
        if not ok:
            failures += 1

    total = len(dates)
    success = total - failures
    print(f"Done. Success: {success}/{total}. Output dir: {extr_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
