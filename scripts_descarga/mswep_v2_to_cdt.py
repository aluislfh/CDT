#!/usr/bin/env python3
"""Download MSWEP v2 daily/monthly from Google Drive (via rclone) and write CDT NetCDF.

This script mirrors CDT logic from R/cdtDownloadRFE_mswep.R for daily/monthly:
- downloads source .nc files
- subsets bbox
- writes CDT-compatible NetCDF with variable 'precip'
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import netCDF4 as nc
import numpy as np

from defaults import DEFAULT_BBOX_HELP, DEFAULT_MAXLAT, DEFAULT_MAXLON, DEFAULT_MINLAT, DEFAULT_MINLON
from netcdf_utils import is_valid_cdt_netcdf

DEFAULT_REMOTE = "gdrive"
DEFAULT_DAILY_FOLDER_ID = "1gWoZ2bK2u5osJ8Iw-dvguZ56Kmz2QWrL"
DEFAULT_MONTHLY_FOLDER_ID = "16BS6ezP7AEJPgZ8dA1FH6-IIIZUp8ASE"


@dataclass(frozen=True)
class ProductSpec:
    product: str
    tstep: str
    dataname: str
    out_name_fmt: str
    folder_id: str


PRODUCT_SPECS = {
    "daily": ProductSpec(
        product="daily",
        tstep="daily",
        dataname="MSWEP_Past_v2.8",
        out_name_fmt="mswep_v2.8_{yyyymmdd}.nc",
        folder_id=DEFAULT_DAILY_FOLDER_ID,
    ),
    "monthly": ProductSpec(
        product="monthly",
        tstep="monthly",
        dataname="MSWEP_Past_v2.8",
        out_name_fmt="mswep_v2.8_{yyyymm}.nc",
        folder_id=DEFAULT_MONTHLY_FOLDER_ID,
    ),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MSWEP v2 files from Google Drive with rclone and convert "
            "to CDT-compatible NetCDF."
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
        "--remote",
        default=DEFAULT_REMOTE,
        help="rclone remote name (default: gdrive).",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Google Drive folder ID override. Defaults by product.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Base output directory. Product folder is created inside it.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep downloaded original .nc files in Data_global.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned downloads/conversions without writing files.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_bbox(minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    if not (-180.0 <= minlon <= 180.0 and -180.0 <= maxlon <= 180.0):
        raise ValueError("Longitude must be within [-180, 180].")
    if not (-90.0 <= minlat <= 90.0 and -90.0 <= maxlat <= 90.0):
        raise ValueError("Latitude must be within [-90, 90].")
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("Invalid bbox: min values must be <= max values.")


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


def target_source_filename(spec: ProductSpec, d: date) -> str:
    if spec.tstep == "monthly":
        return d.strftime("%Y%m") + ".nc"

    # CDT uses year + day-of-year for daily source naming.
    return d.strftime("%Y%j") + ".nc"


def target_output_filename(spec: ProductSpec, d: date) -> str:
    if spec.tstep == "monthly":
        return spec.out_name_fmt.format(yyyymm=d.strftime("%Y%m"))
    return spec.out_name_fmt.format(yyyymmdd=d.strftime("%Y%m%d"))


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "command failed")
    return proc.stdout


def list_remote_nc_files(remote: str, folder_id: str) -> Dict[str, str]:
    cmd = [
        "rclone",
        "lsf",
        f"{remote}:",
        "--drive-root-folder-id",
        folder_id,
        "--recursive",
        "--files-only",
        "--include",
        "*.nc",
    ]
    output = run_cmd(cmd)

    mapping: Dict[str, str] = {}
    for line in output.splitlines():
        rel = line.strip()
        if not rel or not rel.endswith(".nc"):
            continue
        base = Path(rel).name
        # Keep first occurrence if there are duplicates by basename.
        if base not in mapping:
            mapping[base] = rel
    return mapping


def download_one(remote: str, folder_id: str, remote_rel_path: str, local_path: Path) -> None:
    cmd = [
        "rclone",
        "copyto",
        f"{remote}:{remote_rel_path}",
        str(local_path),
        "--drive-root-folder-id",
        folder_id,
    ]
    run_cmd(cmd)


def get_precip_var(ds: nc.Dataset):
    if "precipitation" in ds.variables:
        return ds.variables["precipitation"]

    # Fallback for unexpected naming.
    for key in ds.variables:
        if key.lower() in {"precip", "prcp", "rain", "rainfall"}:
            return ds.variables[key]

    raise KeyError("No precipitation variable found in source NetCDF")


def convert_to_cdt_nc(src_file: Path, out_file: Path, minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    with nc.Dataset(src_file, mode="r") as src:
        var = get_precip_var(src)
        lon_name = None
        lat_name = None
        for dim_name in var.dimensions:
            dlow = dim_name.lower()
            if lon_name is None and dlow in {"lon", "longitude", "x"}:
                lon_name = dim_name
            if lat_name is None and dlow in {"lat", "latitude", "y"}:
                lat_name = dim_name

        if lon_name is None or lat_name is None:
            raise ValueError("could not detect lon/lat dimensions in source NetCDF")

        lon = np.array(src.variables[lon_name][:], dtype=np.float64)
        lat = np.array(src.variables[lat_name][:], dtype=np.float64)
        val = np.array(var[:], dtype=np.float64)
        dims = list(var.dimensions)
        shape = tuple(var.shape)

    val = np.squeeze(val)
    if val.ndim != 2:
        raise ValueError(f"unsupported variable dimensions: {dims} {shape}")

    if len(dims) == 2:
        lon_axis = dims.index(lon_name)
        lat_axis = dims.index(lat_name)
    else:
        kept_axes = [i for i, size in enumerate(shape) if size > 1]
        lon_axis = kept_axes.index(dims.index(lon_name))
        lat_axis = kept_axes.index(dims.index(lat_name))

    if not (lon_axis == 0 and lat_axis == 1):
        val = np.transpose(val, (lon_axis, lat_axis))

    # Mirror CDT: order lat ascending and reorder data accordingly.
    oy = np.argsort(lat)
    lat = lat[oy]
    val = val[:, oy]

    ix = (lon >= minlon) & (lon <= maxlon)
    iy = (lat >= minlat) & (lat <= maxlat)

    if not np.any(ix) or not np.any(iy):
        raise ValueError("bbox does not intersect source grid")

    lon = lon[ix]
    lat = lat[iy]
    val = val[np.ix_(ix, iy)]

    val = np.round(val, 2)
    val[np.isnan(val)] = -99.0

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
        precip[:] = val.astype(np.float32)
        precip.units = "mm"
        precip.long_name = "Precipitation estimates"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    spec = PRODUCT_SPECS[args.product]
    folder_id = args.folder_id or spec.folder_id

    if shutil.which("rclone") is None:
        print("ERROR: rclone no esta instalado o no esta en PATH.", file=sys.stderr)
        return 2

    try:
        validate_bbox(args.minlon, args.maxlon, args.minlat, args.maxlat)
        if spec.tstep == "daily":
            if len(args.start) != 8 or len(args.end) != 8:
                raise ValueError("for daily product, --start and --end must be YYYYMMDD")
            dates = list(iter_daily(args.start, args.end))
        else:
            if len(args.start) != 6 or len(args.end) != 6:
                raise ValueError("for monthly product, --start and --end must be YYYYMM")
            dates = list(iter_monthly(args.start, args.end))
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    data_name = f"{spec.dataname}_{spec.tstep}"
    product_dir = Path(args.outdir).expanduser().resolve() / data_name
    orig_dir = product_dir / "Data_global"
    extr_dir = product_dir / "Extracted"
    orig_dir.mkdir(parents=True, exist_ok=True)
    extr_dir.mkdir(parents=True, exist_ok=True)

    pending_dates = []
    for d in dates:
        out_nc = extr_dir / target_output_filename(spec, d)
        if is_valid_cdt_netcdf(out_nc, ["precip"]):
            print(f"SKIP: valid output already exists: {out_nc}")
        else:
            pending_dates.append(d)

    if not pending_dates:
        print(
            f"Done. Converted: {len(dates)}/{len(dates)}. Missing in Drive: 0. "
            f"Failed: 0. Output dir: {extr_dir}"
        )
        return 0

    try:
        remote_map = list_remote_nc_files(args.remote, folder_id)
    except Exception as exc:
        print(f"Failed to list Google Drive files: {exc}", file=sys.stderr)
        return 1

    missing_remote = 0
    failures = 0

    for d in pending_dates:
        src_name = target_source_filename(spec, d)
        out_name = target_output_filename(spec, d)

        if src_name not in remote_map:
            print(f"MISSING in Drive: {src_name}", file=sys.stderr)
            missing_remote += 1
            continue

        remote_rel_path = remote_map[src_name]
        local_src = orig_dir / src_name
        out_nc = extr_dir / out_name

        if args.verbose:
            print(f"Source: {src_name} -> Output: {out_name}")

        if args.dry_run:
            continue

        try:
            download_one(args.remote, folder_id, remote_rel_path, local_src)
            convert_to_cdt_nc(local_src, out_nc, args.minlon, args.maxlon, args.minlat, args.maxlat)
            if args.verbose:
                print(f"Wrote: {out_nc}")
            if not args.keep_original:
                local_src.unlink(missing_ok=True)
        except Exception as exc:
            print(f"FAILED: {src_name}: {exc}", file=sys.stderr)
            failures += 1

    total = len(dates)
    converted = total - missing_remote - failures
    print(
        f"Done. Converted: {converted}/{total}. Missing in Drive: {missing_remote}. "
        f"Failed: {failures}. Output dir: {extr_dir}"
    )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
