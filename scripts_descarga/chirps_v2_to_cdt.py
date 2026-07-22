#!/usr/bin/env python3
"""Download CHIRPSv2 daily/monthly and write CDT-compatible NetCDF files.

This script mirrors key CDT logic in R/cdtDownloadRFE_chirps.R for:
- CHIRPSv2_daily
- CHIRPSv2_monthly
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from defaults import DEFAULT_BBOX_HELP, DEFAULT_MAXLAT, DEFAULT_MAXLON, DEFAULT_MINLAT, DEFAULT_MINLON
from netcdf_utils import is_valid_cdt_netcdf


BASE_URL = "https://data.chc.ucsb.edu/products"


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
        data_name="CHIRPSv2_daily",
        out_name_fmt="chirps_{yyyymmdd}.nc",
    ),
    "monthly": ProductSpec(
        product="monthly",
        tstep="monthly",
        data_name="CHIRPSv2_monthly",
        out_name_fmt="chirps_{yyyymm}.nc",
    ),
}


@dataclass(frozen=True)
class SourceInfo:
    url: str
    source_type: str  # tif or bil


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CHIRPS v2 from UCSB and convert to CDT-compatible NetCDF files."
        )
    )
    parser.add_argument(
        "--product",
        required=True,
        choices=sorted(PRODUCT_SPECS.keys()),
        help="CHIRPS temporal product.",
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
        "--outdir",
        default=".",
        help="Base output directory. A product folder is created inside it.",
    )
    parser.add_argument(
        "--chirps-global",
        action="store_true",
        help=(
            "Force global source even when bbox falls inside regional subsets. "
            "Equivalent to CDT chirps.global = TRUE."
        ),
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original downloaded compressed files.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep temporary extracted tif/bil files.",
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


def validate_bbox(minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    if not (-180.0 <= minlon <= 180.0 and -180.0 <= maxlon <= 180.0):
        raise ValueError("Longitude must be within [-180, 180].")
    if not (-90.0 <= minlat <= 90.0 and -90.0 <= maxlat <= 90.0):
        raise ValueError("Latitude must be within [-90, 90].")
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("Invalid bbox: min values must be <= max values.")


def region_flags(minlon: float, maxlon: float, minlat: float, maxlat: float) -> Tuple[bool, bool]:
    africa = (
        minlon >= -19.975
        and maxlon <= 54.975
        and minlat >= -39.975
        and maxlat <= 39.975
    )
    camer_carib = (
        minlon >= -92.975
        and maxlon <= -57.025
        and minlat >= 6.025
        and maxlat <= 23.475
    )
    return africa, camer_carib


def build_sources_for_date(
    spec: ProductSpec,
    dt: str,
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
    chirps_global: bool,
) -> List[SourceInfo]:
    africa, camer_carib = region_flags(minlon, maxlon, minlat, maxlat)

    if spec.tstep == "daily":
        year = dt[0:4]
        month = dt[4:6]
        day = dt[6:8]

        if africa and not chirps_global:
            # CDT: CHIRPS-2.0/africa_daily/bils/p05/<year>/v2p0chirpsYYYYMMDD.tar.gz
            fname = f"v2p0chirps{year}{month}{day}.tar.gz"
            return [
                SourceInfo(
                    url=f"{BASE_URL}/CHIRPS-2.0/africa_daily/bils/p05/{year}/{fname}",
                    source_type="bil",
                )
            ]

        # CDT: CHIRPS-2.0/global_daily/tifs/p05/<year>/chirps-v2.0.YYYY.MM.DD.tif.gz
        fname = f"chirps-v2.0.{year}.{month}.{day}.tif.gz"
        return [
            SourceInfo(
                url=f"{BASE_URL}/CHIRPS-2.0/global_daily/tifs/p05/{year}/{fname}",
                source_type="tif",
            )
        ]

    # monthly
    year = dt[0:4]
    month = dt[4:6]
    fname = f"v2p0chirps{year}{month}.tar.gz"

    if africa and not chirps_global:
        # CDT: CHIRPS-2.0/africa_monthly/bils/v2p0chirpsYYYYMM.tar.gz
        return [
            SourceInfo(
                url=f"{BASE_URL}/CHIRPS-2.0/africa_monthly/bils/{fname}",
                source_type="bil",
            )
        ]

    if camer_carib and not chirps_global:
        # CDT: CHIRPS-2.0/camer-carib_monthly/bils/v2p0chirpsYYYYMM.tar.gz
        return [
            SourceInfo(
                url=f"{BASE_URL}/CHIRPS-2.0/camer-carib_monthly/bils/{fname}",
                source_type="bil",
            )
        ]

    # CDT: CHIRPS-2.0/global_monthly/bils/v2p0chirpsYYYYMM.tar.gz
    return [
        SourceInfo(
            url=f"{BASE_URL}/CHIRPS-2.0/global_monthly/bils/{fname}",
            source_type="bil",
        )
    ]


def curl_download(url: str, dest: Path) -> bool:
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


def gunzip_file(src_gz: Path, dst: Path) -> None:
    with gzip.open(src_gz, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def find_one_by_ext(root: Path, ext: str) -> Path:
    matches = sorted(root.rglob(f"*{ext}"))
    if not matches:
        raise FileNotFoundError(f"No {ext} file found in {root}")
    return matches[0]


def decompress_input(src: Path, workdir: Path, source_type: str) -> Path:
    name = src.name.lower()

    if name.endswith(".tar.gz"):
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(path=workdir)
        return find_one_by_ext(workdir, ".bil" if source_type == "bil" else ".tif")

    if name.endswith(".tar"):
        with tarfile.open(src, "r:") as tf:
            tf.extractall(path=workdir)
        return find_one_by_ext(workdir, ".bil" if source_type == "bil" else ".tif")

    if name.endswith(".gz") and not name.endswith(".tar.gz"):
        out_name = src.name[:-3]
        out_path = workdir / out_name
        gunzip_file(src, out_path)
        return out_path

    return src


def extract_to_cdt_nc(
    raster_path: Path,
    out_nc: Path,
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
) -> None:
    with rasterio.open(raster_path) as ds:
        win = from_bounds(minlon, minlat, maxlon, maxlat, transform=ds.transform)
        win = win.round_offsets().round_lengths()

        if win.width <= 0 or win.height <= 0:
            raise ValueError("bbox does not intersect raster extent")

        arr = ds.read(1, window=win, boundless=False)
        transform = ds.window_transform(win)

    if arr.size == 0:
        raise ValueError("empty raster subset")

    # Build center coordinates for cropped window.
    cols = np.arange(arr.shape[1], dtype=np.float64)
    rows = np.arange(arr.shape[0], dtype=np.float64)
    lon = transform.c + (cols + 0.5) * transform.a
    lat_desc = transform.f + (rows + 0.5) * transform.e

    # Mirror CDT chirps.extract.tif_bil transformation:
    # z <- as.matrix(rc); z <- t(z)[, rev(seq_along(y))]
    z = np.transpose(arr)
    z = z[:, ::-1]

    # Latitudes written in ascending order in CDT output.
    lat = np.sort(lat_desc)

    # Mirror CDT cleaning: negative values -> NA, then fill with -99.
    z = z.astype(np.float64)
    z[z < 0] = np.nan
    z[np.isnan(z)] = -99.0

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
            fill_value=np.float32(-99.0),
        )
        precip[:] = z.astype(np.float32)
        precip.units = "mm"
        precip.long_name = "Rainfall Estimate"


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
    original_dir: Path,
    extracted_dir: Path,
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
    chirps_global: bool,
    keep_original: bool,
    keep_extracted: bool,
    verbose: bool,
) -> bool:
    out_name = build_output_name(spec, dt)
    out_nc = outdir / out_name

    if is_valid_cdt_netcdf(out_nc, ["precip"]):
        print(f"SKIP: valid output already exists: {out_nc}")
        return True

    sources = build_sources_for_date(
        spec,
        dt,
        minlon,
        maxlon,
        minlat,
        maxlat,
        chirps_global,
    )

    for src in sources:
        download_name = Path(src.url).name
        orig_file = original_dir / download_name

        if verbose:
            print(f"Downloading: {src.url}")

        if not curl_download(src.url, orig_file):
            continue

        try:
            extracted = decompress_input(orig_file, extracted_dir, src.source_type)
            extract_to_cdt_nc(extracted, out_nc, minlon, maxlon, minlat, maxlat)
            if verbose:
                print(f"Wrote: {out_nc}")
        except Exception as exc:
            print(f"FAILED: {dt} from {src.url}: {exc}", file=sys.stderr)
            return False
        finally:
            if not keep_original and orig_file.exists():
                orig_file.unlink(missing_ok=True)

            if not keep_extracted:
                for item in extracted_dir.iterdir():
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)

        return True

    print(f"FAILED: could not download {dt}", file=sys.stderr)
    return False


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    spec = PRODUCT_SPECS[args.product]

    try:
        validate_bbox(args.minlon, args.maxlon, args.minlat, args.maxlat)
        dates = build_target_dates(spec, args.start, args.end)
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    product_dir = Path(args.outdir).expanduser().resolve() / spec.data_name
    product_dir.mkdir(parents=True, exist_ok=True)

    # Keep a structure close to CDT: Data_* and Extracted.
    original_dir = product_dir / "Data_ucsb"
    extracted_dir = product_dir / "Extracted"
    original_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for dt in dates:
        ok = download_one(
            spec=spec,
            dt=dt,
            outdir=product_dir,
            original_dir=original_dir,
            extracted_dir=extracted_dir,
            minlon=args.minlon,
            maxlon=args.maxlon,
            minlat=args.minlat,
            maxlat=args.maxlat,
            chirps_global=args.chirps_global,
            keep_original=args.keep_original,
            keep_extracted=args.keep_extracted,
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
