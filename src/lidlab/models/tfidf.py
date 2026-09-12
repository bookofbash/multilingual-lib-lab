from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lidlab.labels import normalize_language
from lidlab.models.base import LidModel
from lidlab.schema import Example, ModelCard, Prediction


class TfidfLid(LidModel):
    """Character n-gram baseline. This is the honest closed-set control.

    Trained only on ``data/seed.jsonl``. It exists to show what a linguist-friendly
    sklearn stack can do before any neural weights are downloaded.
    """

    def __init__(self, pipeline: Pipeline, supported: frozenset[str]) -> None:
        self.pipeline = pipeline
        self._card = ModelCard(
            name="tfidf",
            kind="sklearn-char-ngram",
            supported=supported,
            notes="LinearSVC over character 2–5 grams, trained on the builtin seed set.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def train(cls, seed: Sequence[Example]) -> TfidfLid:
        if not seed:
            raise ValueError("tfidf model needs seed examples")
        texts = [item.text for item in seed]
        labels = [normalize_language(item.gold) for item in seed]
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        min_df=1,
                        lowercase=False,
                    ),
                ),
                ("clf", LinearSVC(C=1.0)),
            ]
        )
        pipeline.fit(texts, labels)
        return cls(pipeline, frozenset(labels))

    def predict_one(self, text: str) -> Prediction:
        label = str(self.pipeline.predict([text])[0])
        decision = self.pipeline.decision_function([text])
        confidence = float(decision.max()) if getattr(decision, "max", None) else 0.0
        return Prediction(language=label, confidence=confidence, raw_label=label)
