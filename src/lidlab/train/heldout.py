"""Keep the challenge set out of any training loop."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from lidlab.schema import Example

Row = tuple[str, str]


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def challenge_text_index(examples: Iterable[Example]) -> frozenset[str]:
    return frozenset(normalize_text(item.text) for item in examples)


def filter_training_rows(rows: Sequence[Row], blocked: frozenset[str]) -> tuple[list[Row], int]:
    """Drop rows whose text matches a challenge item. Labels are untouched."""
    kept: list[Row] = []
    dropped = 0
    for text, label in rows:
        if normalize_text(text) in blocked:
            dropped += 1
            continue
        kept.append((text, label))
    return kept, dropped
