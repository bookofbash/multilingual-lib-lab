from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "challenge.jsonl").exists():
            return parent
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate the lidlab repo root (expected data/challenge.jsonl).")


def data_dir() -> Path:
    env = os.environ.get("LIDLAB_DATA")
    if env:
        return Path(env)
    return repo_root() / "data"


def cache_dir() -> Path:
    env = os.environ.get("LIDLAB_CACHE")
    if env:
        path = Path(env)
    else:
        path = Path.home() / ".cache" / "lidlab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def xlmrft_dir() -> Path:
    env = os.environ.get("LIDLAB_XLMRFT")
    if env:
        return Path(env)
    return cache_dir() / "xlmrft"


def reports_dir() -> Path:
    path = repo_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path
