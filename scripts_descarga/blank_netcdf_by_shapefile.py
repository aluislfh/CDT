#!/usr/bin/env python3
"""Blank NetCDF grids outside a reference polygon shapefile.

This script mirrors CDT blanking logic (R/cdtBlankNetCDF_Procs.R and
R/cdtBlanking_Options_functions.R):
- build a polygon mask over grid centers
- optional buffer around polygons
- set values outside polygons to each variable missing value
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import shapely


LON_CANDIDATES = {"lon", "longitude", "x"}
LAT_CANDIDATES = {"lat", "latitude", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Blank NetCDF values outside polygons from a shapefile, "
            "writing new CDT-compatible NetCDF files."
        )
    )
    p.add_argument("--input-dir", required=True, help="Input folder containing .nc files.")
    p.add_argument("--output-dir", required=True, help="Output folder for blanked .nc files.")
    p.add_argument("--shapefile", required=True, help="Polygon shapefile path.")
    p.add_argument(
        "--buffer-option",
        choices=["default", "user"],
        default="user",
        help="CDT-like buffer option. default=4*grid_resolution, user=buffer_width.",
    )
    p.add_argument(
        "--buffer-width",
        type=float,
        default=0.0,
        help="Buffer width in decimal degrees when buffer-option=user.",
    )
    p.add_argument(
        "--var",
        default=None,
        help="Optional variable name to blank. If omitted, blanks all variables that use lon+lat dims.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search .nc files recursively under input-dir.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def list_nc_files(input_dir: Path, recursive: bool) -> List[Path]:
    patt = "**/*.nc" if recursive else "*.nc"
    return sorted(input_dir.glob(patt))


def find_lon_lat_dims(ds: nc.Dataset) -> Tuple[str, str, np.ndarray, np.ndarray]:
    lon_name = None
    lat_name = None

    for name, var in ds.variables.items():
        low = name.lower()
        std = str(getattr(var, "standard_name", "")).lower()
        if lon_name is None and (low in LON_CANDIDATES or std == "longitude") and var.ndim == 1:
            lon_name = name
        if lat_name is None and (low in LAT_CANDIDATES or std == "latitude") and var.ndim == 1:
            lat_name = name

    if lon_name is None or lat_name is None:
        # Fallback from dimension names.
        for dname in ds.dimensions.keys():
            low = dname.lower()
            if lon_name is None and low in LON_CANDIDATES and dname in ds.variables:
                lon_name = dname
            if lat_name is None and low in LAT_CANDIDATES and dname in ds.variables:
                lat_name = dname

    if lon_name is None or lat_name is None:
        raise RuntimeError("Unable to identify lon/lat coordinates")

    lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
    lat = np.array(ds.variables[lat_name][:], dtype=np.float64)
    return lon_name, lat_name, lon, lat


def build_mask_from_shapefile(shp_path: Path, lon: np.ndarray, lat: np.ndarray, buffer_option: str, buffer_width: float) -> np.ndarray:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise RuntimeError("Shapefile has no geometries")

    # Keep only polygonal geometries.
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise RuntimeError("Shapefile must contain Polygon/MultiPolygon geometries")

    geom = gdf.geometry.union_all()

    if lon.size < 2 or lat.size < 2:
        raise RuntimeError("Need at least 2 lon and 2 lat points to compute grid resolution")

    width_lon = (np.nanmax(lon) - np.nanmin(lon)) / (lon.size - 1)
    width_lat = (np.nanmax(lat) - np.nanmin(lat)) / (lat.size - 1)
    width = min(width_lon, width_lat)

    # CDT simplifies polygons before masking.
    geom = shapely.simplify(geom, tolerance=width / 4.0, preserve_topology=True)

    if buffer_option == "default":
        buffer = 4.0 * width
    else:
        buffer = float(buffer_width)

    if buffer > 0:
        geom = shapely.buffer(geom, distance=buffer)

    xx, yy = np.meshgrid(lon, lat, indexing="ij")
    pts = shapely.points(xx.ravel(), yy.ravel())
    inside = shapely.intersects(pts, geom)

    mask = inside.reshape((lon.size, lat.size))
    return mask


def get_fill_value(var: nc.Variable) -> float:
    if hasattr(var, "_FillValue"):
        return float(var._FillValue)
    if hasattr(var, "missing_value"):
        mv = var.missing_value
        if isinstance(mv, np.ndarray):
            return float(mv.flat[0])
        return float(mv)
    return -99.0


def blank_data_array(data: np.ndarray, dims: Tuple[str, ...], lon_dim: str, lat_dim: str, mask: np.ndarray, fill_value: float) -> np.ndarray:
    lon_axis = dims.index(lon_dim)
    lat_axis = dims.index(lat_dim)

    arr = np.array(data, dtype=np.float64)
    perm = [lon_axis, lat_axis] + [i for i in range(arr.ndim) if i not in (lon_axis, lat_axis)]
    arr = np.transpose(arr, perm)

    mask_f = np.where(mask, 1.0, np.nan)
    expand_shape = (slice(None), slice(None)) + (None,) * (arr.ndim - 2)
    arr = arr * mask_f[expand_shape]
    arr[~np.isfinite(arr)] = fill_value

    inv = np.argsort(perm)
    arr = np.transpose(arr, inv)
    return arr


def copy_and_blank_file(src_file: Path, dst_file: Path, shp_path: Path, buffer_option: str, buffer_width: float, only_var: str | None, verbose: bool) -> None:
    with nc.Dataset(src_file, mode="r") as src:
        lon_dim, lat_dim, lon, lat = find_lon_lat_dims(src)
        mask = build_mask_from_shapefile(shp_path, lon, lat, buffer_option, buffer_width)

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(dst_file, mode="w", format="NETCDF4") as dst:
            # Dimensions
            for dname, dim in src.dimensions.items():
                dst.createDimension(dname, (len(dim) if not dim.isunlimited() else None))

            # Global attributes
            for aname in src.ncattrs():
                dst.setncattr(aname, src.getncattr(aname))

            for vname, var in src.variables.items():
                fill_arg = None
                if hasattr(var, "_FillValue"):
                    fill_arg = var.getncattr("_FillValue")

                if fill_arg is None:
                    out_var = dst.createVariable(vname, var.datatype, var.dimensions, zlib=True, complevel=4)
                else:
                    out_var = dst.createVariable(vname, var.datatype, var.dimensions, fill_value=fill_arg, zlib=True, complevel=4)

                for aname in var.ncattrs():
                    if aname == "_FillValue":
                        continue
                    out_var.setncattr(aname, var.getncattr(aname))

                data = var[:]
                can_blank = (lon_dim in var.dimensions and lat_dim in var.dimensions)
                should_blank = can_blank and (only_var is None or only_var == vname)

                if should_blank:
                    fv = get_fill_value(var)
                    arr = blank_data_array(data, var.dimensions, lon_dim, lat_dim, mask, fv)
                    out_var[:] = arr.astype(var.datatype)
                    if verbose:
                        print(f"Blanked variable {vname} in {src_file.name}")
                else:
                    out_var[:] = data


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    shp_path = Path(args.shapefile).expanduser().resolve()

    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}")
        return 2
    if not shp_path.exists():
        print(f"Shapefile not found: {shp_path}")
        return 2

    files = list_nc_files(input_dir, args.recursive)
    if not files:
        print("No .nc files found in input directory")
        return 1

    failed = 0
    for src_file in files:
        rel = src_file.relative_to(input_dir)
        dst_file = output_dir / rel
        try:
            copy_and_blank_file(
                src_file=src_file,
                dst_file=dst_file,
                shp_path=shp_path,
                buffer_option=args.buffer_option,
                buffer_width=args.buffer_width,
                only_var=args.var,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"Wrote: {dst_file}")
        except Exception as exc:
            failed += 1
            print(f"FAILED {src_file}: {exc}")

    total = len(files)
    ok = total - failed
    print(f"Done. Success: {ok}/{total}. Output dir: {output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
