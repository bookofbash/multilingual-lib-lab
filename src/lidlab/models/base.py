from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lidlab.schema import ModelCard, Prediction


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
