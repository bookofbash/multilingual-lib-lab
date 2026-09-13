"""XLM-R fine-tuned on WiLI (challenge set held out)."""

from __future__ import annotations

from lidlab.labels import normalize_prediction
from lidlab.models.base import LidModel, pipeline_rows
from lidlab.paths import cache_dir, xlmrft_dir
from lidlab.schema import ModelCard, Prediction


class XlmrFtLid(LidModel):
    """Classifier head trained by `lidlab train-xlmr`. Wider than papluca's 20."""

    def __init__(self, pipeline, supported: frozenset[str], path) -> None:
        self.pipeline = pipeline
        self._card = ModelCard(
            name="xlmrft",
            kind="xlm-roberta-wili",
            supported=supported,
            notes=f"Fine-tuned xlm-roberta-base on WiLI-2018; checkpoint {path}. Challenge JSONL was held out.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def load(cls) -> XlmrFtLid:
        path = xlmrft_dir()
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"no xlmrft checkpoint at {path}; run: uv run lidlab train-xlmr"
            )
        try:
            from transformers import AutoConfig, pipeline
        except ImportError as exc:
            raise ImportError("neural extra required: pip install 'lidlab[neural]'") from exc

        config = AutoConfig.from_pretrained(path)
        supported = frozenset(normalize_prediction(str(label))[0] for label in config.id2label.values())
        pipe = pipeline(
            "text-classification",
            model=str(path),
            tokenizer=str(path),
            top_k=1,
            truncation=True,
            model_kwargs={"cache_dir": str(cache_dir() / "hf")},
        )
        return cls(pipe, supported, path)

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
