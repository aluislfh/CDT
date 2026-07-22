"""Small validation helpers shared by resumable download scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import netCDF4 as nc
import numpy as np


def is_valid_cdt_netcdf(path: Path, required_vars: Sequence[str]) -> bool:
    """Return True only for a readable, populated CDT lon/lat NetCDF file."""
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with nc.Dataset(path, mode="r") as ds:
            if "lon" not in ds.dimensions or "lat" not in ds.dimensions:
                return False
            if "lon" not in ds.variables or "lat" not in ds.variables:
                return False

            shape = (len(ds.dimensions["lon"]), len(ds.dimensions["lat"]))
            if shape[0] == 0 or shape[1] == 0:
                return False

            for name in required_vars:
                if name not in ds.variables:
                    return False
                var = ds.variables[name]
                if tuple(var.shape) != shape:
                    return False
                if np.ma.count(var[:]) == 0:
                    return False
    except (OSError, RuntimeError, ValueError):
        return False

    return True
