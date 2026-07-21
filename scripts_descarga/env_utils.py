from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    txt = line.strip()
    if not txt or txt.startswith("#") or "=" not in txt:
        return None

    key, value = txt.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]

    return key, value


def find_env_file(start: Path | None = None) -> Path | None:
    roots = []
    if start is not None:
        roots.append(start.resolve())
    roots.append(Path.cwd().resolve())
    roots.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    for root in roots:
        for base in [root, *root.parents]:
            if base in seen:
                continue
            seen.add(base)
            env_path = base / ".env"
            if env_path.exists() and env_path.is_file():
                return env_path
    return None


def load_dotenv(start: Path | None = None) -> Path | None:
    env_path = find_env_file(start)
    if env_path is None:
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)

    return env_path