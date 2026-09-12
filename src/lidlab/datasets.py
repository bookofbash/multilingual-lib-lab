from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from lidlab.paths import data_dir
from lidlab.schema import Example


def load_jsonl(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            examples.append(Example.from_dict(raw))
    return examples


def load_challenge(path: Path | None = None) -> list[Example]:
    return load_jsonl(path or data_dir() / "challenge.jsonl")


def load_seed(path: Path | None = None) -> list[Example]:
    return load_jsonl(path or data_dir() / "seed.jsonl")


def validate_examples(examples: Iterable[Example]) -> list[str]:
    """Return human-readable problems; empty means the set is structurally sound."""
    from lidlab.labels import parse_gold

    problems: list[str] = []
    seen: set[str] = set()
    for example in examples:
        if not example.id:
            problems.append("example is missing id")
            continue
        if example.id in seen:
            problems.append(f"duplicate id: {example.id}")
        seen.add(example.id)
        if not example.text.strip():
            problems.append(f"{example.id}: empty text")
        try:
            spec = parse_gold(example.gold)
        except ValueError as exc:
            problems.append(f"{example.id}: {exc}")
            continue
        if example.matrix and example.matrix not in spec.all_of and example.matrix not in spec.any_of:
            problems.append(f"{example.id}: matrix {example.matrix!r} is not in gold {example.gold!r}")
    return problems
