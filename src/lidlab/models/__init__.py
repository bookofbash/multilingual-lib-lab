from __future__ import annotations

from collections.abc import Sequence

from lidlab.schema import Example, ModelCard
from lidlab.models.base import LidModel

MODEL_KINDS = ("tfidf", "fasttext", "glotlid", "xlmr", "embed")


def build_model(name: str, seed: Sequence[Example] | None = None) -> LidModel:
    key = name.strip().lower()
    if key == "tfidf":
        from lidlab.models.tfidf import TfidfLid

        return TfidfLid.train(seed or [])
    if key == "fasttext":
        from lidlab.models.fasttext_lid import FastTextLid

        return FastTextLid.load()
    if key == "glotlid":
        from lidlab.models.glotlid import GlotLid

        return GlotLid.load()
    if key == "xlmr":
        from lidlab.models.xlmr import XlmrLid

        return XlmrLid.load()
    if key == "embed":
        from lidlab.models.embed import EmbedLid

        return EmbedLid.train(seed or [])
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_KINDS)}")


def card_for(model: LidModel) -> ModelCard:
    return model.card
