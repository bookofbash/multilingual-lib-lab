"""XLM-R fine-tune on a public LID set. The challenge JSONL is never a source."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lidlab.datasets import load_challenge
from lidlab.labels import normalize_language
from lidlab.paths import cache_dir, xlmrft_dir
from lidlab.train.heldout import Row, challenge_text_index, filter_training_rows

WILI_DATASET = "MartinThoma/wili_2018"
BASE_MODEL = "xlm-roberta-base"

# papluca's 20, plus challenge languages that head cannot name.
# WiLI will not have every code (yue is `zh-yue`; some dialects are absent).
TARGET_LANGUAGES = frozenset(
    {
        "am",
        "ar",
        "arz",
        "bg",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "eu",
        "fr",
        "ga",
        "haw",
        "hi",
        "hr",
        "id",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sk",
        "sr",
        "sv",
        "sw",
        "th",
        "tr",
        "ur",
        "vi",
        "yo",
        "yue",
        "zh",
    }
)


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    n: int
    languages: tuple[str, ...]
    dropped_heldout: int
    missing_targets: tuple[str, ...]


def _wili_name(dataset, row: dict) -> str:
    label = row["label"]
    feature = dataset.features["label"]
    names = getattr(feature, "names", None)
    if names is not None and isinstance(label, int):
        return str(names[label])
    return str(label)


def load_wili_rows(dataset_id: str = WILI_DATASET, split: str = "train") -> list[Row]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("train extra required: uv sync --extra train") from exc

    dataset = load_dataset(dataset_id, split=split, cache_dir=str(cache_dir() / "hf"))
    rows: list[Row] = []
    for row in dataset:
        raw = _wili_name(dataset, row)
        language = normalize_language(raw)
        if language not in TARGET_LANGUAGES:
            continue
        text = str(row["sentence"]).strip()
        if not text:
            continue
        rows.append((text, language))
    return rows


def subsample(rows: Sequence[Row], max_per_lang: int, seed: int) -> list[Row]:
    by_lang: dict[str, list[str]] = defaultdict(list)
    for text, language in rows:
        by_lang[language].append(text)
    rng = random.Random(seed)
    out: list[Row] = []
    for language in sorted(by_lang):
        texts = list(by_lang[language])
        rng.shuffle(texts)
        for text in texts[:max_per_lang]:
            out.append((text, language))
    rng.shuffle(out)
    return out


def prepare_rows(
    rows: Sequence[Row],
    *,
    blocked: frozenset[str],
    max_per_lang: int,
    seed: int,
    limit: int | None = None,
) -> tuple[list[Row], int]:
    kept, dropped = filter_training_rows(rows, blocked)
    kept = subsample(kept, max_per_lang, seed)
    if limit is not None:
        kept = kept[:limit]
    return kept, dropped


def run_train(
    *,
    dataset_id: str = WILI_DATASET,
    base_model: str = BASE_MODEL,
    max_per_lang: int = 400,
    epochs: int = 2,
    batch_size: int = 16,
    seed: int = 0,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> TrainResult:
    if "challenge.jsonl" in dataset_id:
        raise ValueError("refusing to train on challenge.jsonl")

    blocked = challenge_text_index(load_challenge())
    raw_rows = load_wili_rows(dataset_id)
    present = {language for _, language in raw_rows}
    missing = tuple(sorted(TARGET_LANGUAGES - present))
    rows, dropped = prepare_rows(
        raw_rows,
        blocked=blocked,
        max_per_lang=max_per_lang,
        seed=seed,
        limit=limit,
    )
    if not rows:
        raise RuntimeError("no training rows after held-out and language filters")

    output = output_dir or xlmrft_dir()
    output.mkdir(parents=True, exist_ok=True)
    _fit(rows, base_model=base_model, epochs=epochs, batch_size=batch_size, seed=seed, output=output)

    languages = tuple(sorted({language for _, language in rows}))
    meta = {
        "dataset": dataset_id,
        "base_model": base_model,
        "n": len(rows),
        "languages": list(languages),
        "max_per_lang": max_per_lang,
        "epochs": epochs,
        "dropped_heldout": dropped,
        "missing_targets": list(missing),
        "held_out": "data/challenge.jsonl",
    }
    (output / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return TrainResult(
        output_dir=output,
        n=len(rows),
        languages=languages,
        dropped_heldout=dropped,
        missing_targets=missing,
    )


def _select_training_kwargs(accepted: set[str], requested: dict) -> dict:
    """Keep TrainingArguments keys that this transformers version actually has."""
    out = dict(requested)
    if "warmup_ratio" in out and "warmup_ratio" not in accepted and "warmup_steps" in accepted:
        out["warmup_steps"] = out.pop("warmup_ratio")
    if "eval_strategy" in out and "eval_strategy" not in accepted and "evaluation_strategy" in accepted:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    if "evaluation_strategy" in out and "evaluation_strategy" not in accepted and "eval_strategy" in accepted:
        out["eval_strategy"] = out.pop("evaluation_strategy")
    return {key: value for key, value in out.items() if key in accepted}


def _training_args(requested: dict):
    import inspect

    from transformers import TrainingArguments

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    accepted.discard("self")
    return TrainingArguments(**_select_training_kwargs(accepted, requested))


def _fit(
    rows: list[Row],
    *,
    base_model: str,
    epochs: int,
    batch_size: int,
    seed: int,
    output: Path,
) -> None:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
        )
    except ImportError as exc:
        raise ImportError("train extra required: uv sync --extra train") from exc

    labels = sorted({language for _, language in rows})
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=str(cache_dir() / "hf"))

    texts = [text for text, _ in rows]
    y = [label2id[language] for _, language in rows]
    split_at = max(1, int(len(rows) * 0.9)) if len(rows) >= 20 else len(rows)
    train_ds = Dataset.from_dict({"text": texts[:split_at], "label": y[:split_at]})
    eval_ds = None
    if split_at < len(rows):
        eval_ds = Dataset.from_dict({"text": texts[split_at:], "label": y[split_at:]})

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch["text"], truncation=True, max_length=256)
        encoded["labels"] = batch["label"]
        return encoded

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text", "label"])
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(cache_dir() / "hf"),
    )
    use_fp16 = bool(torch.cuda.is_available())
    run_dir = output / "runs"
    common = dict(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        save_total_limit=1,
        fp16=use_fp16,
        report_to="none",
        seed=seed,
    )
    if eval_ds is None:
        common.update(save_strategy="epoch", eval_strategy="no")
    else:
        common.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    args = _training_args(common)

    trainer_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
