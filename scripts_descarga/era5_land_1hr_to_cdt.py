#!/usr/bin/env python3
"""Download ERA5-Land hourly data from CDS and write CDT-compatible NetCDF."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

import netCDF4 as nc
import numpy as np

from env_utils import load_dotenv


CDS_API_BASE = "https://cds.climate.copernicus.eu/api"
CDS_RESOURCE = "reanalysis-era5-land"
@dataclass
class VarOption:
    cdt_var: str
    api_var: List[str]
    varid: List[str]
    nc_name: List[str]
    nc_longname: List[str]
    nc_units: List[str]
    units_fun: List[str]
    units_args: List[str]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ERA5-Land 1-hour variables and convert to CDT NetCDF."
    )
    p.add_argument("--start", required=True, help="YYYYMMDDHH")
    p.add_argument("--end", required=True, help="YYYYMMDDHH")
    p.add_argument("--minlon", type=float, required=True)
    p.add_argument("--maxlon", type=float, required=True)
    p.add_argument("--minlat", type=float, required=True)
    p.add_argument("--maxlat", type=float, required=True)
    p.add_argument(
        "--variables",
        required=True,
        help="Comma-separated CDT variable keys from era5_Land_options.csv (e.g. evp,pet,tair).",
    )
    p.add_argument(
        "--cds-token",
        default=None,
        help="CDS API token. Defaults to CDS_TOKEN from environment or .env.",
    )
    p.add_argument("--outdir", default=".")
    p.add_argument("--poll-seconds", type=int, default=3)
    p.add_argument("--max-polls", type=int, default=400)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def validate_bbox(minlon: float, maxlon: float, minlat: float, maxlat: float) -> None:
    if not (-180.0 <= minlon <= 180.0 and -180.0 <= maxlon <= 180.0):
        raise ValueError("Longitude must be within [-180, 180]")
    if not (-90.0 <= minlat <= 90.0 and -90.0 <= maxlat <= 90.0):
        raise ValueError("Latitude must be within [-90, 90]")
    if minlon > maxlon or minlat > maxlat:
        raise ValueError("Invalid bbox")


def load_era5_land_options(csv_path: Path) -> Dict[str, VarOption]:
    groups: Dict[str, Dict[str, List[str]]] = {}
    current = None

    with csv_path.open("r", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for r in rows:
            cdt_var = (r.get("cdt_var") or "").strip()
            if cdt_var:
                current = cdt_var
                groups[current] = {
                    "api_var": [],
                    "varid": [],
                    "nc_name": [],
                    "nc_longname": [],
                    "nc_units": [],
                    "units_fun": [],
                    "units_args": [],
                }
            if not current:
                continue

            if (r.get("api_var") or "").strip():
                groups[current]["api_var"].append(r["api_var"].strip())
            if (r.get("varid") or "").strip():
                groups[current]["varid"].append(r["varid"].strip())
            if (r.get("nc_name") or "").strip():
                groups[current]["nc_name"].append(r["nc_name"].strip())
            if (r.get("nc_longname") or "").strip():
                groups[current]["nc_longname"].append(r["nc_longname"].strip())
            if (r.get("nc_units") or "").strip():
                groups[current]["nc_units"].append(r["nc_units"].strip())
            groups[current]["units_fun"].append((r.get("units_fun") or "").strip())
            groups[current]["units_args"].append((r.get("units_args") or "").strip())

    out: Dict[str, VarOption] = {}
    for k, v in groups.items():
        out[k] = VarOption(
            cdt_var=k,
            api_var=v["api_var"],
            varid=v["varid"],
            nc_name=v["nc_name"],
            nc_longname=v["nc_longname"],
            nc_units=v["nc_units"],
            units_fun=v["units_fun"],
            units_args=v["units_args"],
        )
    return out


def split_by_day(start: datetime, end: datetime) -> Dict[str, List[datetime]]:
    out: Dict[str, List[datetime]] = {}
    cur = start
    while cur <= end:
        k = cur.strftime("%Y%m%d")
        out.setdefault(k, []).append(cur)
        cur += timedelta(hours=1)
    return out


def curl_json(method: str, url: str, token: str, payload: Dict | None = None) -> Dict:
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "-X",
        method,
        url,
        "-H",
        f"PRIVATE-TOKEN: {token}",
        "-H",
        "Content-Type: application/json",
        "-w",
        "\n%{http_code}",
    ]

    if payload is not None:
        cmd.extend(["--data", json.dumps(payload)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "curl request failed")

    out = proc.stdout
    idx = out.rfind("\n")
    if idx < 0:
        raise RuntimeError("unexpected curl output")

    body = out[:idx].strip()
    code_txt = out[idx + 1 :].strip()
    try:
        code = int(code_txt)
    except ValueError as exc:
        raise RuntimeError(f"invalid HTTP status from curl: {code_txt}") from exc

    if code >= 400:
        raise RuntimeError(f"HTTP {code}: {body}")

    if not body:
        return {}

    return json.loads(body)


def send_request(token: str, request_inputs: Dict) -> Dict:
    url = f"{CDS_API_BASE}/retrieve/v1/processes/{CDS_RESOURCE}/execute"
    return curl_json("POST", url, token, payload={"inputs": request_inputs})


def get_href_links(js: Dict) -> List[str]:
    links = js.get("links") or []
    hrefs = []
    for item in links:
        h = item.get("href")
        if h:
            hrefs.append(h)
    return hrefs


def get_link_by_rel(js: Dict, rel: str) -> str | None:
    links = js.get("links") or []
    for item in links:
        if item.get("rel") == rel and item.get("href"):
            return item.get("href")
    return None


def poll_task(token: str, task_url: str, poll_seconds: int, max_polls: int) -> Dict:
    url = task_url
    for _ in range(max_polls):
        js = curl_json("GET", url, token)
        status = js.get("status")

        if status == "successful":
            return js
        if status == "failed":
            raise RuntimeError(json.dumps(js, ensure_ascii=False))

        monitor = get_link_by_rel(js, "monitor")
        hrefs = get_href_links(js)
        if monitor:
            url = monitor
        elif hrefs:
            url = hrefs[0]

        time.sleep(poll_seconds)

    raise TimeoutError("CDS task polling exceeded max polls")


def get_asset_url(js: Dict, token: str) -> str:
    asset = js.get("asset") or {}
    value = asset.get("value") or {}
    href = value.get("href")
    if href:
        return href

    hrefs = get_href_links(js)
    for h in hrefs:
        try:
            jj = curl_json("GET", h, token)
        except Exception:
            continue
        a = (jj.get("asset") or {}).get("value") or {}
        ah = a.get("href")
        if ah:
            return ah

    raise RuntimeError("Unable to get download URL from CDS response")


def download_file(url: str, dest: Path) -> None:
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
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "download failed")


def apply_convert(data: np.ndarray, op: str, arg: str) -> np.ndarray:
    if not op or op == "":
        return data
    x = float(arg)
    if op == "-":
        return data - x
    if op == "/":
        return data / x
    if op == "*":
        return data * x
    return data


def write_cdt_nc(path: Path, lon: np.ndarray, lat: np.ndarray, fields: List[np.ndarray], names: List[str], units: List[str], longnames: List[str]) -> None:
    with nc.Dataset(path, mode="w", format="NETCDF4") as dst:
        dst.createDimension("lon", lon.size)
        dst.createDimension("lat", lat.size)

        vlon = dst.createVariable("lon", "f4", ("lon",))
        vlat = dst.createVariable("lat", "f4", ("lat",))
        vlon[:] = lon.astype(np.float32)
        vlat[:] = lat.astype(np.float32)
        vlon.units = "degrees_east"
        vlon.long_name = "Longitude"
        vlat.units = "degrees_north"
        vlat.long_name = "Latitude"

        for i, arr in enumerate(fields):
            arr = np.round(arr, 3)
            arr[np.isnan(arr)] = -9999
            vv = dst.createVariable(
                names[i],
                "f4",
                ("lon", "lat"),
                zlib=True,
                complevel=9,
                fill_value=np.float32(-9999),
            )
            vv[:] = arr.astype(np.float32)
            vv.units = units[i]
            vv.long_name = longnames[i]


def format_cdt_from_source(src_nc: Path, out_dir: Path, cdt_var: str, opt: VarOption) -> None:
    with nc.Dataset(src_nc, mode="r") as ds:
        lon = np.array(ds.variables["longitude"][:], dtype=np.float64)
        lat = np.array(ds.variables["latitude"][:], dtype=np.float64)
        time_vals = np.array(ds.variables["valid_time"][:], dtype=np.float64)
        time_units = ds.variables["valid_time"].units

        data_vars: Dict[str, np.ndarray] = {}
        for v in opt.varid:
            var = ds.variables[v]
            arr = np.array(var[:], dtype=np.float64)
            dims = list(var.dimensions)

            # Normalize to [lon, lat, time] like CDT expects downstream.
            try:
                ax_lon = dims.index("longitude")
                ax_lat = dims.index("latitude")
                if "valid_time" in dims:
                    ax_time = dims.index("valid_time")
                elif "time" in dims:
                    ax_time = dims.index("time")
                else:
                    raise ValueError
            except ValueError as exc:
                raise RuntimeError(f"Unexpected dimensions for variable {v}: {dims}") from exc

            arr = np.transpose(arr, (ax_lon, ax_lat, ax_time))
            data_vars[v] = arr

    ox = np.argsort(lon)
    oy = np.argsort(lat)
    lon = lon[ox]
    lat = lat[oy]

    for v in opt.varid:
        x = data_vars[v]
        x = x[ox, :, :]
        x = x[:, oy, :]
        data_vars[v] = x

    # Build timestamps from units like "hours since 1900-01-01 00:00:00"
    t0 = nc.num2date(time_vals, units=time_units)

    for j, dt in enumerate(t0):
        fields: List[np.ndarray] = []
        for i, v in enumerate(opt.varid):
            arr = data_vars[v][:, :, j]
            arr[~np.isfinite(arr)] = np.nan
            arr = apply_convert(arr, opt.units_fun[i], opt.units_args[i])
            fields.append(arr)

        ts = dt.strftime("%Y%m%d%H")
        out_file = out_dir / f"{cdt_var}_{ts}.nc"
        write_cdt_nc(out_file, lon, lat, fields, opt.nc_name, opt.nc_units, opt.nc_longname)


def main(argv: Sequence[str]) -> int:
    load_dotenv(Path(__file__).resolve().parent)
    args = parse_args(argv)

    if not args.cds_token:
        args.cds_token = os.environ.get("CDS_TOKEN")
    if not args.cds_token:
        print(
            "CDS token is required. Use --cds-token or CDS_TOKEN in environment or .env.",
            file=sys.stderr,
        )
        return 2

    try:
        validate_bbox(args.minlon, args.maxlon, args.minlat, args.maxlat)
        start = datetime.strptime(args.start, "%Y%m%d%H")
        end = datetime.strptime(args.end, "%Y%m%d%H")
        if end < start:
            raise ValueError("end must be >= start")
    except Exception as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    options_path = Path(__file__).resolve().parents[1] / "inst" / "cdt" / "reanalysis" / "era5_Land_options.csv"
    opts = load_era5_land_options(options_path)

    req_vars = [x.strip() for x in args.variables.split(",") if x.strip()]
    missing = [v for v in req_vars if v not in opts]
    if missing:
        print(f"Unknown variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    if "hum" in req_vars:
        print("Variable 'hum' is not supported yet in this script.", file=sys.stderr)
        return 2

    area = f"{args.maxlat}/{args.minlon}/{args.minlat}/{args.maxlon}"
    by_day = split_by_day(start, end)

    base_out = Path(args.outdir).expanduser().resolve() / "ERA5_1Hr_Land"
    base_out.mkdir(parents=True, exist_ok=True)

    failures = 0

    for cdt_var in req_vars:
        opt = opts[cdt_var]
        var_out = base_out / f"ERA5_Land_{cdt_var}"
        var_out.mkdir(parents=True, exist_ok=True)

        for ymd, times in by_day.items():
            req = {
                "dataset_short_name": CDS_RESOURCE,
                "product_type": "reanalysis",
                "data_format": "netcdf",
                "download_format": "unarchived",
                "variable": opt.api_var,
                "area": area,
                "year": ymd[0:4],
                "month": ymd[4:6],
                "day": ymd[6:8],
                "time": [t.strftime("%H:%M") for t in times],
            }

            if args.verbose:
                print(f"Requesting {cdt_var} for {ymd} ({len(times)} hours)")

            try:
                js = send_request(args.cds_token, req)

                status = js.get("status")
                if status == "failed":
                    raise RuntimeError(json.dumps(js, ensure_ascii=False))

                monitor = get_link_by_rel(js, "monitor")
                hrefs = get_href_links(js)
                task_url = monitor if monitor else (hrefs[0] if hrefs else None)
                if not task_url:
                    raise RuntimeError("CDS response missing task links")

                task_final = poll_task(args.cds_token, task_url, args.poll_seconds, args.max_polls)
                dl_url = get_asset_url(task_final, args.cds_token)

                tmp_nc = var_out / f"tmp_{cdt_var}_{ymd}.nc"
                download_file(dl_url, tmp_nc)
                format_cdt_from_source(tmp_nc, var_out, cdt_var, opt)
                tmp_nc.unlink(missing_ok=True)
            except Exception as exc:
                print(f"FAILED: {cdt_var} {ymd}: {exc}", file=sys.stderr)
                failures += 1

    if failures == 0:
        print(f"Done. Output dir: {base_out}")
        return 0

    print(f"Done with failures: {failures}. Output dir: {base_out}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
