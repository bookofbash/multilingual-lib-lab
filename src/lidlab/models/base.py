from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lidlab.schema import ModelCard, Prediction


def top_labels(classes: Sequence[str], scores, k: int = 2) -> tuple[str, tuple[str, ...], float]:
    """Return ``(top_label, alternatives, top_score)`` from a 1-d score vector."""
    import numpy as np

    values = np.asarray(scores, dtype=float).reshape(-1)
    names = [str(name) for name in classes]
    order = np.argsort(values)[::-1][: max(k, 1)]
    top = names[int(order[0])]
    rest = tuple(names[int(index)] for index in order[1:] if names[int(index)] != top)
    return top, rest, float(values[int(order[0])])


def pipeline_rows(output) -> list[dict]:
    """Normalize transformers pipeline output to a list of ``{label, score}`` rows."""
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    return [row for row in (output or []) if isinstance(row, dict)]


class LidModel(ABC):
    @property
    @abstractmethod
    def card(self) -> ModelCard:
        raise NotImplementedError

    @abstractmethod
    def predict_one(self, text: str) -> Prediction:
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return [self.predict_one(text) for text in texts]
