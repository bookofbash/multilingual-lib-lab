from __future__ import annotations

from collections.abc import Sequence

from sklearn.linear_model import LogisticRegression

from lidlab.labels import normalize_language
from lidlab.models.base import LidModel, top_labels
from lidlab.paths import cache_dir
from lidlab.schema import Example, ModelCard, Prediction

CHECKPOINT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbedLid(LidModel):
    """Sentence embedding + linear classifier. Closed-set, seed-trained."""

    def __init__(self, encoder, classifier: LogisticRegression, supported: frozenset[str]) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self._card = ModelCard(
            name="embed",
            kind="embedding-classifier",
            supported=supported,
            notes=f"{CHECKPOINT} + multinomial logistic regression on the builtin seed set.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def train(cls, seed: Sequence[Example]) -> EmbedLid:
        if not seed:
            raise ValueError("embed model needs seed examples")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("neural extra required: pip install 'lidlab[neural]'") from exc

        encoder = SentenceTransformer(CHECKPOINT, cache_folder=str(cache_dir() / "hf"))
        texts = [item.text for item in seed]
        labels = [normalize_language(item.gold) for item in seed]
        vectors = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(vectors, labels)
        return cls(encoder, classifier, frozenset(labels))

    def predict_one(self, text: str) -> Prediction:
        vector = self.encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
        if hasattr(self.classifier, "predict_proba"):
            scores = self.classifier.predict_proba(vector)
        else:
            scores = self.classifier.decision_function(vector)
        language, alternatives, confidence = top_labels(self.classifier.classes_, scores, k=2)
        extras = {"alternatives": alternatives} if alternatives else {}
        return Prediction(
            language=language,
            confidence=confidence,
            raw_label=language,
            extras=extras,
        )
