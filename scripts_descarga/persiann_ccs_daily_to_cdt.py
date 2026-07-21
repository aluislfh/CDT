#!/usr/bin/env python3
"""Wrapper for PERSIANN-CCS daily download + CDT conversion."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    main_script = Path(__file__).with_name("persiann_cdr_ccs_to_cdt.py")
    sys.argv = [str(main_script), "--source", "ccs", "--tstep", "daily", *sys.argv[1:]]
    runpy.run_path(str(main_script), run_name="__main__")
