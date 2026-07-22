#!/usr/bin/env python3
"""Download PERSIANN-CDR/CCS daily or monthly and write CDT-compatible NetCDF."""

from __future__ import annotations

import argparse
import gzip
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
from netcdf_utils import is_valid_cdt_netcdf

BASE_URL = "https://persiann.eng.uci.edu/CHRSdata"
MISSVAL = -9999.0


@dataclass(frozen=True)
class SourceSpec:
    source: str
    tstep: str
    dirpath: str
    fileformat: str
    dateformat: str
    nc_prefix: str
    endian: str
    factor: float
    lon0: float
    lat0: float
    nlon: int
    nlat: int
    res: float


SPECS = {
    ("cdr", "daily"): SourceSpec(
        source="cdr",
        tstep="daily",
        dirpath="PERSIANN-CDR/daily",
        fileformat="aB1_d{yy}{doy}.bin.gz",
        dateformat="%y%j",
        nc_prefix="persiann-cdr",
        endian="little",
        factor=1.0,
        lon0=0.125,
        lat0=59.875,
        nlon=1440,
        nlat=480,
        res=0.25,
    ),
    ("cdr", "monthly"): SourceSpec(
        source="cdr",
        tstep="monthly",
        dirpath="PERSIANN-CDR/mthly",
        fileformat="aB1_m{yy}{mm}.bin.gz",
        dateformat="%y%m",
        nc_prefix="persiann-cdr",
        endian="little",
        factor=1.0,
        lon0=0.125,
        lat0=59.875,
        nlon=1440,
        nlat=480,
        res=0.25,
    ),
    ("ccs", "daily"): SourceSpec(
        source="ccs",
        tstep="daily",
        dirpath="PERSIANN-CCS/daily",
        fileformat="rgccs1d{yy}{doy}.bin.gz",
        dateformat="%y%j",
        nc_prefix="persiann-css",
        endian="big",
        factor=1.0,
        lon0=0.02,
        lat0=59.98,
        nlon=9000,
        nlat=3000,
        res=0.04,
    ),
    ("ccs", "monthly"): SourceSpec(
        source="ccs",
        tstep="monthly",
        dirpath="PERSIANN-CCS/mthly",
        fileformat="rgccs1m{yy}{mm}.bin.gz",
        dateformat="%y%m",
        nc_prefix="persiann-css",
        endian="big",
        factor=1.0,
        lon0=0.02,
        lat0=59.98,
        nlon=9000,
        nlat=3000,
        res=0.04,
    ),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PERSIANN-CDR/CCS and convert to CDT NetCDF."
    )
    p.add_argument("--source", required=True, choices=["cdr", "ccs"])
    p.add_argument("--tstep", required=True, choices=["daily", "monthly"])
    p.add_argument("--start", required=True, help="YYYYMMDD (daily) or YYYYMM (monthly)")
    p.add_argument("--end", required=True, help="YYYYMMDD (daily) or YYYYMM (monthly)")
    p.add_argument("--minlon", type=float, default=DEFAULT_MINLON, help=DEFAULT_BBOX_HELP)
    p.add_argument("--maxlon", type=float, default=DEFAULT_MAXLON, help=DEFAULT_BBOX_HELP)
    p.add_argument("--minlat", type=float, default=DEFAULT_MINLAT, help=DEFAULT_BBOX_HELP)
    p.add_argument("--maxlat", type=float, default=DEFAULT_MAXLAT, help=DEFAULT_BBOX_HELP)
    p.add_argument("--outdir", default=".")
    p.add_argument("--keep-original", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def iter_daily(start: str, end: str) -> Iterable[date]:
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    if d1 < d0:
        raise ValueError("end date must be >= start date")
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def iter_monthly(start: str, end: str) -> Iterable[date]:
    d0 = datetime.strptime(start, "%Y%m").date().replace(day=1)
    d1 = datetime.strptime(end, "%Y%m").date().replace(day=1)
    if d1 < d0:
        raise ValueError("end date must be >= start date")
    cur = d0
    while cur <= d1:
        yield cur
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = date(year, month, 1)


def validate_bbox(minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    if not (-180.0 <= minlon <= 180.0 and -180.0 <= maxlon <= 180.0):
        raise ValueError("Longitude must be within [-180, 180]")
    if not (-90.0 <= minlat <= 90.0 and -90.0 <= maxlat <= 90.0):
        raise ValueError("Latitude must be within [-90, 90]")
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("Invalid bbox")


def src_filename(spec: SourceSpec, d: date) -> str:
    yy = d.strftime("%y")
    mm = d.strftime("%m")
    doy = d.strftime("%j")
    return spec.fileformat.format(yy=yy, mm=mm, doy=doy)


def out_filename(spec: SourceSpec, d: date) -> str:
    if spec.tstep == "daily":
        return f"{spec.nc_prefix}_{d.strftime('%Y%m%d')}.nc"
    return f"{spec.nc_prefix}_{d.strftime('%Y%m')}.nc"


def download(url: str, dest: Path) -> bool:
    cmd = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--output",
        str(dest),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


def read_bin_from_gz(gz_file: Path, spec: SourceSpec) -> np.ndarray:
    with gzip.open(gz_file, "rb") as f:
        raw = f.read()

    dtype = np.dtype("<f4" if spec.endian == "little" else ">f4")
    arr = np.frombuffer(raw, dtype=dtype)

    expected = spec.nlon * spec.nlat
    if arr.size < expected:
        raise ValueError(f"Invalid binary length in {gz_file.name}")

    arr = arr[:expected].astype(np.float64)
    arr[arr == MISSVAL] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def convert_to_cdt_nc(src_gz: Path, out_nc: Path, spec: SourceSpec, minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    val = read_bin_from_gz(src_gz, spec)

    lon = np.arange(spec.lon0, spec.lon0 + spec.res * spec.nlon, spec.res, dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = np.arange(spec.lat0, spec.lat0 - spec.res * spec.nlat, -spec.res, dtype=np.float64)

    ox = np.argsort(lon)
    oy = np.argsort(lat)
    lon = lon[ox]
    lat = lat[oy]

    val = np.reshape(val, (spec.nlon, spec.nlat), order="F")
    val = val[ox, :][:, oy]

    ix = (lon >= minlon) & (lon <= maxlon)
    iy = (lat >= minlat) & (lat <= maxlat)

    if not np.any(ix) or not np.any(iy):
        raise ValueError("bbox does not intersect source grid")

    lon = lon[ix]
    lat = lat[iy]
    val = val[np.ix_(ix, iy)]

    val = np.round(val * spec.factor, 1)
    val[np.isnan(val)] = MISSVAL

    with nc.Dataset(out_nc, mode="w", format="NETCDF4") as dst:
        dst.createDimension("lon", lon.size)
        dst.createDimension("lat", lat.size)

        lon_var = dst.createVariable("lon", "f4", ("lon",))
        lat_var = dst.createVariable("lat", "f4", ("lat",))
        lon_var[:] = lon.astype(np.float32)
        lat_var[:] = lat.astype(np.float32)
        lon_var.units = "degreeE"
        lon_var.long_name = "Longitude"
        lat_var.units = "degreeN"
        lat_var.long_name = "Latitude"

        precip = dst.createVariable(
            "precip",
            "f4",
            ("lon", "lat"),
            zlib=True,
            complevel=9,
            fill_value=np.float32(MISSVAL),
        )
        precip[:] = val.astype(np.float32)
        precip.units = "mm"
        precip.long_name = (
            "Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks"
        )


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    try:
        validate_bbox(args.minlon, args.maxlon, args.minlat, args.maxlat)
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    spec = SPECS[(args.source, args.tstep)]

    try:
        dates = list(iter_daily(args.start, args.end)) if spec.tstep == "daily" else list(iter_monthly(args.start, args.end))
    except Exception as exc:
        print(f"Invalid date range: {exc}", file=sys.stderr)
        return 2

    data_name = f"{'PERSIANN-CDR' if args.source == 'cdr' else 'PERSIANN-CCS'}_{args.tstep}"
    product_dir = Path(args.outdir).expanduser().resolve() / data_name
    orig_dir = product_dir / "Data_global"
    extr_dir = product_dir / "Extracted"
    orig_dir.mkdir(parents=True, exist_ok=True)
    extr_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for d in dates:
        sname = src_filename(spec, d)
        oname = out_filename(spec, d)

        url = f"{BASE_URL}/{spec.dirpath}/{sname}"
        local_src = orig_dir / sname
        out_nc = extr_dir / oname

        if is_valid_cdt_netcdf(out_nc, ["precip"]):
            print(f"SKIP: valid output already exists: {out_nc}")
            continue

        if args.verbose:
            print(f"Downloading: {url}")

        if not download(url, local_src):
            print(f"FAILED: download {sname}", file=sys.stderr)
            failures += 1
            continue

        try:
            convert_to_cdt_nc(local_src, out_nc, spec, args.minlon, args.maxlon, args.minlat, args.maxlat)
            if args.verbose:
                print(f"Wrote: {out_nc}")
        except Exception as exc:
            print(f"FAILED: convert {sname}: {exc}", file=sys.stderr)
            failures += 1
        finally:
            if not args.keep_original:
                local_src.unlink(missing_ok=True)

    total = len(dates)
    success = total - failures
    print(f"Done. Success: {success}/{total}. Output dir: {extr_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
