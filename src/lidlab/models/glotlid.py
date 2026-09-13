from __future__ import annotations

from pathlib import Path

from lidlab.labels import normalize_prediction
from lidlab.models.base import LidModel
from lidlab.paths import cache_dir
from lidlab.schema import ModelCard, Prediction


def _supported_from_labels(raw_labels: list[str]) -> frozenset[str]:
    return frozenset(normalize_prediction(raw)[0] for raw in raw_labels)


class GlotLid(LidModel):
    """GlotLID v3: 2000+ language-script labels, including a long tail."""

    def __init__(self, model, path: Path) -> None:
        supported = _supported_from_labels(list(model.labels))
        self.model = model
        self._card = ModelCard(
            name="glotlid",
            kind="glotlid-v3",
            supported=supported,
            size_bytes=path.stat().st_size if path.exists() else None,
            notes="cis-lmu/glotlid model.bin (v3). Best coverage of the comparison set; slower than lid.176.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def load(cls) -> GlotLid:
        try:
            import fasttext
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("fasttext extra required: pip install 'lidlab[fasttext]'") from exc

        path = Path(
            hf_hub_download(
                repo_id="cis-lmu/glotlid",
                filename="model.bin",
                cache_dir=str(cache_dir() / "hf"),
            )
        )
        model = fasttext.load_model(str(path))
        return cls(model, path)

    def predict_one(self, text: str) -> Prediction:
        cleaned = text.replace("\n", " ").strip() or " "
        labels, probs = self.model.predict(cleaned, k=2)
        raw = labels[0]
        language, script = normalize_prediction(raw)
        alternatives = []
        for alt_raw in labels[1:]:
            alt_lang, _ = normalize_prediction(alt_raw)
            if alt_lang != language:
                alternatives.append(alt_lang)
        extras = {"alternatives": tuple(alternatives)} if alternatives else {}
        return Prediction(
            language=language,
            confidence=float(probs[0]),
            raw_label=raw,
            script=script,
            extras=extras,
        )
