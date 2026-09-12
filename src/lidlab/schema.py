from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    """One labeled LID item.

    ``gold`` uses a small notation layered on ISO-like codes:

    - ``ja`` — single language
    - ``ja+en`` — code-switching; both languages are present
    - ``de|no|da|sv|nl`` — any listed label is acceptable (homograph / ambiguity)
    - ``zh`` plus ``script`` / ``region`` for finer analysis, not extra gold classes
    """

    id: str
    text: str
    gold: str
    phenomena: tuple[str, ...] = ()
    script: str | None = None
    region: str | None = None
    matrix: str | None = None
    note: str = ""
    split: str = "challenge"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Example:
        phenomena = raw.get("phenomena") or []
        if isinstance(phenomena, str):
            phenomena = [phenomena]
        return cls(
            id=str(raw["id"]),
            text=str(raw["text"]),
            gold=str(raw["gold"]),
            phenomena=tuple(phenomena),
            script=raw.get("script"),
            region=raw.get("region"),
            matrix=raw.get("matrix"),
            note=raw.get("note") or "",
            split=raw.get("split") or "challenge",
        )


@dataclass(frozen=True)
class Prediction:
    language: str
    confidence: float
    raw_label: str
    script: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCard:
    name: str
    kind: str
    supported: frozenset[str]
    size_bytes: int | None = None
    notes: str = ""
