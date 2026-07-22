#!/usr/bin/env python3
"""Run CDT download/processing scripts declared in an INI namelist."""

from __future__ import annotations

import argparse
import configparser
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


JOB_PREFIX = "job:"
GLOBAL_SECTION = "global"
AUTO_OPTIONS = [
    ("start", "--start"),
    ("end", "--end"),
    ("minlon", "--minlon"),
    ("maxlon", "--maxlon"),
    ("minlat", "--minlat"),
    ("maxlat", "--maxlat"),
    ("outdir", "--outdir"),
]


@dataclass
class Job:
    name: str
    description: str
    enabled: bool
    command: List[str]
    working_dir: Path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read a namelist INI file and execute the configured CDT scripts."
    )
    p.add_argument(
        "--namelist",
        default="download_namelist.ini",
        help="Path to the INI namelist file.",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated job names to run, for example gpm_early_daily,era5_land_pet.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining jobs after a failure.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum number of enabled products to run simultaneously "
            "(default: 1, sequential namelist order)."
        ),
    )
    return p.parse_args(argv)


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    low = value.strip().lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def get_value(cfg: configparser.ConfigParser, section: str, key: str) -> str | None:
    if cfg.has_option(section, key):
        value = cfg.get(section, key).strip()
        return value if value else None
    if section != GLOBAL_SECTION and cfg.has_option(GLOBAL_SECTION, key):
        value = cfg.get(GLOBAL_SECTION, key).strip()
        return value if value else None
    return None


def get_flag_state(cfg: configparser.ConfigParser, section: str, key: str, default: bool) -> bool:
    value = get_value(cfg, section, key)
    return parse_bool(value, default=default)


def resolve_script_path(base_dir: Path, scripts_dir: str | None, script_value: str) -> Path:
    candidate = Path(script_value)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        search_roots = [base_dir, Path.cwd().resolve()]
        for root in search_roots:
            candidates.append(root / candidate)
            if scripts_dir:
                candidates.append(root / scripts_dir / candidate)

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"script not found: {script_value}")


def option_present(extra_args: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in extra_args)


def build_job(cfg: configparser.ConfigParser, section: str, base_dir: Path) -> Job:
    name = section[len(JOB_PREFIX) :]
    description = get_value(cfg, section, "description") or name
    enabled = get_flag_state(cfg, section, "enabled", default=False)

    script_value = get_value(cfg, section, "script")
    if script_value is None:
        raise ValueError(f"missing script in section [{section}]")

    python_cmd = get_value(cfg, section, "python") or "python3"
    scripts_dir = get_value(cfg, section, "scripts_dir")
    script_path = resolve_script_path(base_dir, scripts_dir, script_value)

    extra_args = shlex.split(get_value(cfg, section, "args") or "")
    command = [python_cmd, str(script_path)]

    use_dates = get_flag_state(cfg, section, "use_dates", default=True)
    use_bbox = get_flag_state(cfg, section, "use_bbox", default=True)
    use_outdir = get_flag_state(cfg, section, "use_outdir", default=True)
    use_verbose = get_flag_state(cfg, section, "use_verbose", default=True)

    for key, option in AUTO_OPTIONS:
        if key in {"start", "end"} and not use_dates:
            continue
        if key in {"minlon", "maxlon", "minlat", "maxlat"} and not use_bbox:
            continue
        if key == "outdir" and not use_outdir:
            continue

        value = get_value(cfg, section, key)
        if value is None or option_present(extra_args, option):
            continue
        command.extend([option, value])

    if use_verbose and get_flag_state(cfg, section, "verbose", default=False) and not option_present(extra_args, "--verbose"):
        command.append("--verbose")

    command.extend(extra_args)

    working_dir = base_dir
    working_dir_value = get_value(cfg, section, "working_dir")
    if working_dir_value is not None:
        working_dir = (base_dir / working_dir_value).resolve()

    return Job(
        name=name,
        description=description,
        enabled=enabled,
        command=command,
        working_dir=working_dir,
    )


def load_jobs(namelist_path: Path) -> tuple[configparser.ConfigParser, List[Job]]:
    cfg = configparser.ConfigParser(interpolation=None)
    with namelist_path.open("r", encoding="utf-8") as f:
        cfg.read_file(f)

    jobs = []
    for section in cfg.sections():
        if section.startswith(JOB_PREFIX):
            jobs.append(build_job(cfg, section, namelist_path.parent.resolve()))
    return cfg, jobs


def format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def select_jobs(jobs: Iterable[Job], only: str) -> List[Job]:
    if not only.strip():
        return list(jobs)
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    return [job for job in jobs if job.name in wanted]


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.jobs < 1:
        print("--jobs must be greater than or equal to 1.", file=sys.stderr)
        return 2

    namelist_path = Path(args.namelist).expanduser().resolve()
    if not namelist_path.is_file():
        print(f"Namelist not found: {namelist_path}", file=sys.stderr)
        return 2

    try:
        cfg, jobs = load_jobs(namelist_path)
    except Exception as exc:
        print(f"Failed to read namelist: {exc}", file=sys.stderr)
        return 2

    jobs = select_jobs(jobs, args.only)
    runnable_jobs = [job for job in jobs if job.enabled]
    skipped_jobs = [job for job in jobs if not job.enabled]

    if not jobs:
        print("No jobs found in the namelist.", file=sys.stderr)
        return 2

    stop_on_error = get_flag_state(cfg, GLOBAL_SECTION, "stop_on_error", default=True)
    if args.continue_on_error:
        stop_on_error = False

    print(f"Namelist: {namelist_path}")
    print(f"Jobs selected: {len(jobs)} | enabled: {len(runnable_jobs)} | disabled: {len(skipped_jobs)}")

    for job in jobs:
        if not job.enabled:
            print(f"SKIP [{job.name}] disabled")

    failures = 0
    executed = 0

    if args.dry_run:
        for job in runnable_jobs:
            executed += 1
            print(f"RUN  [{job.name}] {job.description}")
            print(f"CMD  {format_command(job.command)}")
    else:
        pending = list(runnable_jobs)
        running: List[tuple[Job, subprocess.Popen]] = []
        stop_scheduling = False

        while pending or running:
            while pending and len(running) < args.jobs and not stop_scheduling:
                job = pending.pop(0)
                executed += 1
                print(f"RUN  [{job.name}] {job.description}", flush=True)
                print(f"CMD  {format_command(job.command)}", flush=True)
                proc = subprocess.Popen(job.command, cwd=job.working_dir)
                running.append((job, proc))

            completed: List[tuple[Job, subprocess.Popen]] = []
            for job, proc in running:
                if proc.poll() is not None:
                    completed.append((job, proc))

            if not completed:
                if running:
                    time.sleep(0.1)
                    continue
                break

            for job, proc in completed:
                running.remove((job, proc))
                if proc.returncode != 0:
                    failures += 1
                    print(f"FAIL [{job.name}] exit code {proc.returncode}", file=sys.stderr, flush=True)
                    if stop_on_error:
                        stop_scheduling = True
                else:
                    print(f"OK   [{job.name}]", flush=True)

        if stop_scheduling and pending:
            print(
                f"STOP: {len(pending)} enabled job(s) not started after a failure.",
                file=sys.stderr,
            )

    print(
        f"Summary: executed={executed}, failures={failures}, "
        f"dry_run={args.dry_run}, jobs={args.jobs}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
