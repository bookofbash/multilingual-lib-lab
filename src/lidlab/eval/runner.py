from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lidlab.analysis.failures import write_failure_notes
from lidlab.datasets import load_challenge, load_seed
from lidlab.eval.metrics import RunMetrics, score_run
from lidlab.eval.report import write_report
from lidlab.models import MODEL_KINDS, build_model
from lidlab.paths import reports_dir
from lidlab.schema import Example


@dataclass
class EvalResult:
    run_dir: Path
    metrics: list[RunMetrics]


def run_eval(
    model_names: list[str],
    *,
    limit: int | None = None,
    out_dir: Path | None = None,
) -> EvalResult:
    unknown = [name for name in model_names if name not in MODEL_KINDS]
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")

    examples = load_challenge()
    if limit is not None:
        examples = examples[:limit]
    seed = load_seed()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir or (reports_dir() / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[RunMetrics] = []
    for name in model_names:
        model = build_model(name, seed=seed)
        texts = [example.text for example in examples]
        started = time.perf_counter()
        predictions = model.predict(texts)
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_item = elapsed_ms / max(len(examples), 1)
        result = score_run(examples, predictions, model.card, per_item)
        metrics.append(result)
        _write_json(run_dir / f"{name}.json", result.to_json())

    write_report(run_dir / "report.md", examples, metrics)
    write_failure_notes(run_dir / "failures.md", examples, metrics)
    _write_json(
        run_dir / "summary.json",
        {
            "models": [item.model for item in metrics],
            "n": len(examples),
            "exact": {item.model: item.exact for item in metrics},
            "codeswitch_hit": {item.model: item.codeswitch_hit for item in metrics},
            "family": {item.model: item.family for item in metrics},
            "covered_exact": {item.model: item.covered_exact for item in metrics},
            "latency_ms_per_example": {item.model: item.latency_ms_per_example for item in metrics},
        },
    )
    latest = reports_dir() / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve())
    return EvalResult(run_dir=run_dir, metrics=metrics)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def preview_examples(examples: list[Example], n: int = 5) -> list[str]:
    return [f"{item.id}\t{item.gold}\t{item.text}" for item in examples[:n]]
