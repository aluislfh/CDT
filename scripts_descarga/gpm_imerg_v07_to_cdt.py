#!/usr/bin/env python3
"""Download GPM IMERG V07 subsets and write CDT-compatible NetCDF files.

This script mirrors CDT logic implemented in R/cdtDownloadRFE_gpm_imerg.R
for:
- GPM_L3_IMERG_V07_EARLY_daily
- GPM_L3_IMERG_V07_FINAL_daily
- GPM_L3_IMERG_V07_FINAL_monthly
"""

from __future__ import annotations

import argparse
import calendar
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from env_utils import load_dotenv


OPENDAP_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap"
LEVEL = "GPM_L3"
VAR_ID = "precipitation"
LON_GRID = np.arange(-179.9, 180.0, 0.1)
LAT_GRID = np.arange(-89.9, 90.0, 0.1)


@dataclass(frozen=True)
class ProductSpec:
    product: str
    data_name: str
    dataset: str
    out_name_fmt: str
    kind: str
    tstep: str


PRODUCT_SPECS = {
    "early_daily": ProductSpec(
        product="early_daily",
        data_name="GPM_L3_IMERG_V07_EARLY_daily",
        dataset="GPM_3IMERGDE.07",
        out_name_fmt="imerg_early_{yyyymmdd}.nc",
        kind="EARLY",
        tstep="daily",
    ),
    "final_daily": ProductSpec(
        product="final_daily",
        data_name="GPM_L3_IMERG_V07_FINAL_daily",
        dataset="GPM_3IMERGDF.07",
        out_name_fmt="imerg_final_{yyyymmdd}.nc",
        kind="FINAL",
        tstep="daily",
    ),
    "final_monthly": ProductSpec(
        product="final_monthly",
        data_name="GPM_L3_IMERG_V07_FINAL_monthly",
        dataset="GPM_3IMERGM.07",
        out_name_fmt="imerg_final_{yyyymm}.nc",
        kind="FINAL",
        tstep="monthly",
    ),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GPM IMERG V07 data from GES DISC OPeNDAP and convert "
            "to CDT-compatible NetCDF files."
        )
    )
    parser.add_argument(
        "--product",
        required=True,
        choices=sorted(PRODUCT_SPECS.keys()),
        help="Product to download.",
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
    parser.add_argument("--minlon", type=float, required=True)
    parser.add_argument("--maxlon", type=float, required=True)
    parser.add_argument("--minlat", type=float, required=True)
    parser.add_argument("--maxlat", type=float, required=True)
    parser.add_argument(
        "--outdir",
        default=".",
        help="Base output directory. A product folder is created inside it.",
    )
    parser.add_argument(
        "--earthdata-user",
        default=os.environ.get("EARTHDATA_USERNAME"),
        help="NASA Earthdata username. Defaults to EARTHDATA_USERNAME.",
    )
    parser.add_argument(
        "--earthdata-password",
        default=os.environ.get("EARTHDATA_PASSWORD"),
        help="NASA Earthdata password. Defaults to EARTHDATA_PASSWORD.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep downloaded temporary subset files.",
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


def get_bbox_subset(minlon: float, maxlon: float, minlat: float, maxlat: float) -> Tuple[str, str]:
    ix = np.where((LON_GRID >= minlon) & (LON_GRID <= maxlon))[0]
    iy = np.where((LAT_GRID >= minlat) & (LAT_GRID <= maxlat))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError("bbox does not intersect IMERG global grid")

    ilon_start = max(int(ix.min()) - 1, 0)
    ilon_end = int(ix.max())
    ilat_start = max(int(iy.min()) - 1, 0)
    ilat_end = int(iy.max())

    sublon = f"[{ilon_start}:1:{ilon_end}]"
    sublat = f"[{ilat_start}:1:{ilat_end}]"
    return sublon, sublat


def build_daily_source_candidates(spec: ProductSpec, yyyymmdd: str) -> List[str]:
    year = yyyymmdd[0:4]
    month = yyyymmdd[4:6]
    day_name = f"3B-DAY"
    kind = "-E" if spec.product == "early_daily" else ""

    if spec.product == "final_daily":
        filenames = [
            f"{day_name}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4",
            f"{day_name}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07C.nc4",
            f"{day_name}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07C.nc4.nc4",
        ]
    else:
        filenames = [
            f"{day_name}{kind}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07B.nc4",
            f"{day_name}{kind}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07C.nc4.nc4",
            f"{day_name}{kind}.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07C.nc4",
        ]

    base = f"{OPENDAP_BASE}/{LEVEL}/{spec.dataset}/{year}/{month}"
    return [f"{base}/{filename}" for filename in filenames]


def build_monthly_source_url(spec: ProductSpec, yyyymm: str) -> str:
    year = yyyymm[0:4]
    month = yyyymm[4:6]
    yyyymmdd = f"{yyyymm}01"
    filename = (
        f"3B-MO.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.{month}.V07B.HDF5.nc4"
    )
    return f"{OPENDAP_BASE}/{LEVEL}/{spec.dataset}/{year}/{filename}"


def add_subset_query(url: str, sublon: str, sublat: str) -> str:
    request = f"{VAR_ID}[0:1:0]{sublon}{sublat},lon{sublon},lat{sublat}"
    return f"{url}?{request}"


def write_netrc(path: Path, user: str, password: str) -> None:
    path.write_text(
        "\n".join(
            [
                "machine urs.earthdata.nasa.gov",
                f"  login {user}",
                f"  password {password}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def curl_download(url: str, dest: Path, netrc_path: Path, cookie_path: Path) -> bool:
    cmd = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--globoff",
        "--netrc-file",
        str(netrc_path),
        "--cookie",
        str(cookie_path),
        "--cookie-jar",
        str(cookie_path),
        "--output",
        str(dest),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


def curl_download_text(url: str, netrc_path: Path, cookie_path: Path) -> str | None:
    cmd = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--globoff",
        "--netrc-file",
        str(netrc_path),
        "--cookie",
        str(cookie_path),
        "--cookie-jar",
        str(cookie_path),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_ascii_number_list(line: str) -> np.ndarray:
    _, values = line.split(",", 1)
    return np.array([float(item.strip()) for item in values.split(",")], dtype=np.float64)


def parse_imerg_ascii_subset(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = None
    lat = None
    rows: List[np.ndarray] = []
    pending_lon_row = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Dataset:"):
            continue
        if line.startswith("lon,"):
            lon = parse_ascii_number_list(line)
            continue
        if line.startswith("lat,"):
            lat = parse_ascii_number_list(line)
            continue
        if line.startswith("precipitation.lat,"):
            continue
        if line.startswith("precipitation.precipitation["):
            pending_lon_row = True
            data_part = line.split(",", 1)
            if len(data_part) == 2 and data_part[1].strip():
                rows.append(np.array([float(item.strip()) for item in data_part[1].split(",")], dtype=np.float64))
                pending_lon_row = False
            continue
        if pending_lon_row:
            rows.append(np.array([float(item.strip()) for item in line.split(",")], dtype=np.float64))
            pending_lon_row = False

    if lon is None or lat is None or not rows:
        raise ValueError("could not parse IMERG ASCII subset response")

    prcp = np.vstack(rows)
    if prcp.shape != (lon.size, lat.size):
        raise ValueError(f"unexpected ASCII subset shape: {prcp.shape}, expected {(lon.size, lat.size)}")
    return lon, lat, prcp


def write_cdt_precip_nc(
    out_file: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    prcp: np.ndarray,
    long_name: str,
) -> None:
    prcp = np.round(prcp, 2)
    prcp = np.where(np.isnan(prcp), -99.0, prcp)

    with nc.Dataset(out_file, mode="w", format="NETCDF4") as dst:
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
        precip[:] = prcp.astype(np.float32)
        precip.units = "mm"
        precip.long_name = long_name


def build_ascii_subset_url(url: str, sublon: str, sublat: str) -> str:
    base, _, query = add_subset_query(url, sublon, sublat).partition("?")
    return f"{base}.ascii?{query}"


def download_daily_ascii_fallback(
    source_candidates: List[str],
    out_file: Path,
    sublon: str,
    sublat: str,
    netrc_path: Path,
    cookie_path: Path,
    verbose: bool,
) -> bool:
    long_name = "Precipitation estimates from various precipitation-relevant satellite passive microwave"
    for source in source_candidates:
        ascii_url = build_ascii_subset_url(source, sublon, sublat)
        if verbose:
            print(f"Downloading ASCII fallback: {ascii_url}")
        text = curl_download_text(ascii_url, netrc_path, cookie_path)
        if text is None:
            continue
        try:
            lon, lat, prcp = parse_imerg_ascii_subset(text)
            write_cdt_precip_nc(out_file, lon, lat, prcp, long_name)
            return True
        except Exception:
            continue
    return False


def get_long_name(src: nc.Dataset, var_name: str) -> str:
    var = src.variables[var_name]
    long_name = getattr(var, "LongName", None)
    if long_name:
        return " ".join(str(long_name).replace("\n", " ").split())
    long_name = getattr(var, "long_name", None)
    if long_name and str(long_name) != var_name:
        return str(long_name)
    return "Precipitation estimates from various precipitation-relevant satellite passive microwave"


def convert_to_cdt_nc(src_file: Path, out_file: Path, spec: ProductSpec) -> None:
    with nc.Dataset(src_file, mode="r") as src:
        lon = np.array(src.variables["lon"][:], dtype=np.float64)
        lat = np.array(src.variables["lat"][:], dtype=np.float64)
        prcp = np.array(src.variables[VAR_ID][:], dtype=np.float64)
        long_name = get_long_name(src, VAR_ID)

    if spec.tstep == "monthly":
        yyyymm = out_file.stem.split("_")[-1]
        year = int(yyyymm[0:4])
        month = int(yyyymm[4:6])
        days_in_month = calendar.monthrange(year, month)[1]
        prcp = prcp * 24.0 * days_in_month

    prcp = np.transpose(prcp)
    write_cdt_precip_nc(out_file, lon, lat, prcp, long_name)


def build_target_dates(spec: ProductSpec, start: str, end: str) -> List[str]:
    if spec.tstep == "daily":
        if len(start) != 8 or len(end) != 8:
            raise ValueError("for daily product, --start and --end must be YYYYMMDD")
        return list(iter_daily(start, end))

    if len(start) != 6 or len(end) != 6:
        raise ValueError("for monthly product, --start and --end must be YYYYMM")
    return list(iter_monthly(start, end))


def build_output_name(spec: ProductSpec, dt: str) -> str:
    if spec.tstep == "daily":
        return spec.out_name_fmt.format(yyyymmdd=dt)
    return spec.out_name_fmt.format(yyyymm=dt)


def download_one(
    spec: ProductSpec,
    dt: str,
    outdir: Path,
    sublon: str,
    sublat: str,
    netrc_path: Path,
    cookie_path: Path,
    keep_temp: bool,
    verbose: bool,
) -> bool:
    out_name = build_output_name(spec, dt)
    out_file = outdir / out_name
    tmp_file = outdir / f"tmp_{out_name}"

    if spec.tstep == "daily":
        source_candidates = build_daily_source_candidates(spec, dt)
    else:
        source_candidates = [build_monthly_source_url(spec, dt)]

    ok = False
    for source in source_candidates:
        url = add_subset_query(source, sublon, sublat)
        if verbose:
            print(f"Downloading: {url}")
        if curl_download(url, tmp_file, netrc_path, cookie_path):
            ok = True
            break

    if not ok:
        if spec.tstep == "daily":
            ok = download_daily_ascii_fallback(
                source_candidates,
                out_file,
                sublon,
                sublat,
                netrc_path,
                cookie_path,
                verbose,
            )
        if ok:
            if verbose:
                print(f"Wrote: {out_file}")
            return True
        print(f"FAILED: could not download {dt}", file=sys.stderr)
        return False

    try:
        convert_to_cdt_nc(tmp_file, out_file, spec)
    except Exception as exc:
        print(f"FAILED: conversion error for {dt}: {exc}", file=sys.stderr)
        return False
    finally:
        if not keep_temp and tmp_file.exists():
            tmp_file.unlink(missing_ok=True)

    if verbose:
        print(f"Wrote: {out_file}")
    return True


def main(argv: Sequence[str]) -> int:
    load_dotenv(Path(__file__).resolve().parent)
    args = parse_args(argv)
    spec = PRODUCT_SPECS[args.product]

    if not args.earthdata_user or not args.earthdata_password:
        print(
            "Earthdata credentials are required. Use --earthdata-user/--earthdata-password "
            "or EARTHDATA_USERNAME/EARTHDATA_PASSWORD env vars.",
            file=sys.stderr,
        )
        return 2

    if not (-180.0 <= args.minlon <= 180.0 and -180.0 <= args.maxlon <= 180.0):
        print("Longitude must be within [-180, 180].", file=sys.stderr)
        return 2
    if not (-90.0 <= args.minlat <= 90.0 and -90.0 <= args.maxlat <= 90.0):
        print("Latitude must be within [-90, 90].", file=sys.stderr)
        return 2
    if args.minlon > args.maxlon or args.minlat > args.maxlat:
        print("Invalid bbox: min values must be <= max values.", file=sys.stderr)
        return 2

    try:
        dates = build_target_dates(spec, args.start, args.end)
        sublon, sublat = get_bbox_subset(args.minlon, args.maxlon, args.minlat, args.maxlat)
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    product_dir = Path(args.outdir).expanduser().resolve() / spec.data_name
    product_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="imerg_auth_") as td:
        auth_dir = Path(td)
        netrc_path = auth_dir / ".netrc"
        cookie_path = auth_dir / ".cookies"
        write_netrc(netrc_path, args.earthdata_user, args.earthdata_password)

        failures = 0
        for dt in dates:
            ok = download_one(
                spec=spec,
                dt=dt,
                outdir=product_dir,
                sublon=sublon,
                sublat=sublat,
                netrc_path=netrc_path,
                cookie_path=cookie_path,
                keep_temp=args.keep_temp,
                verbose=args.verbose,
            )
            if not ok:
                failures += 1

    total = len(dates)
    success = total - failures
    print(f"Done. Success: {success}/{total}. Output dir: {product_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
