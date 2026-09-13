from __future__ import annotations

from lidlab.labels import normalize_prediction
from lidlab.models.base import LidModel, pipeline_rows
from lidlab.paths import cache_dir
from lidlab.schema import ModelCard, Prediction

# papluca/xlm-roberta-base-language-detection
XLMR_LANGUAGES = frozenset(
    {
        "ar",
        "bg",
        "de",
        "el",
        "en",
        "es",
        "fr",
        "hi",
        "it",
        "ja",
        "nl",
        "pl",
        "pt",
        "ru",
        "sw",
        "th",
        "tr",
        "ur",
        "vi",
        "zh",
    }
)
CHECKPOINT = "papluca/xlm-roberta-base-language-detection"


class XlmrLid(LidModel):
    """Fine-tuned XLM-R classifier. High accuracy, narrow label set."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._card = ModelCard(
            name="xlmr",
            kind="xlm-roberta-classifier",
            supported=XLMR_LANGUAGES,
            notes=f"{CHECKPOINT}. Strong on its 20 languages; coverage failures are the interesting result.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def load(cls) -> XlmrLid:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("neural extra required: pip install 'lidlab[neural]'") from exc

        pipe = pipeline(
            "text-classification",
            model=CHECKPOINT,
            top_k=1,
            truncation=True,
            model_kwargs={"cache_dir": str(cache_dir() / "hf")},
        )
        return cls(pipe)

    def predict_one(self, text: str) -> Prediction:
        output = self.pipeline(text, top_k=2, truncation=True)
        rows = pipeline_rows(output)
        item = rows[0]
        raw = str(item["label"])
        language, script = normalize_prediction(raw)
        alternatives = []
        for row in rows[1:]:
            alt_lang, _ = normalize_prediction(str(row["label"]))
            if alt_lang != language:
                alternatives.append(alt_lang)
        extras = {"alternatives": tuple(alternatives)} if alternatives else {}
        return Prediction(
            language=language,
            confidence=float(item.get("score") or 0.0),
            raw_label=raw,
            script=script,
            extras=extras,
        )
