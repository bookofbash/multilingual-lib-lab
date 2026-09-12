from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from lidlab.labels import model_covers, parse_gold, same_family
from lidlab.schema import Example, ModelCard, Prediction


@dataclass
class ItemScore:
    example_id: str
    gold: str
    predicted: str
    raw_label: str
    confidence: float
    exact: bool
    codeswitch_hit: bool
    family: bool
    covered: bool
    phenomena: tuple[str, ...]
    note: str


@dataclass
class RunMetrics:
    model: str
    n: int
    exact: float
    codeswitch_hit: float
    family: float
    covered_exact: float
    n_covered: int
    latency_ms_per_example: float
    size_bytes: int | None
    by_phenomenon: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[tuple[str, str, int]] = field(default_factory=list)
    items: list[ItemScore] = field(default_factory=list)

    def to_json(self) -> dict:
        payload = asdict(self)
        return payload


def score_item(example: Example, prediction: Prediction, supported: frozenset[str]) -> ItemScore:
    gold = parse_gold(example.gold)
    predicted = prediction.language
    exact = predicted in gold.any_of
    codeswitch_hit = predicted in gold.all_of
    family = exact or any(same_family(predicted, lang) for lang in gold.any_of)
    covered = model_covers(supported, gold)
    return ItemScore(
        example_id=example.id,
        gold=example.gold,
        predicted=predicted,
        raw_label=prediction.raw_label,
        confidence=prediction.confidence,
        exact=exact,
        codeswitch_hit=codeswitch_hit,
        family=family,
        covered=covered,
        phenomena=example.phenomena,
        note=example.note,
    )


def _mean(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def score_run(
    examples: list[Example],
    predictions: list[Prediction],
    card: ModelCard,
    latency_ms_per_example: float,
) -> RunMetrics:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions length mismatch")
    items = [score_item(example, pred, card.supported) for example, pred in zip(examples, predictions, strict=True)]
    covered = [item for item in items if item.covered]
    codeswitch_items = [item for item in items if "+" in item.gold]
    by_phenomenon: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[ItemScore]] = defaultdict(list)
    for item in items:
        tags = item.phenomena or ("untagged",)
        for tag in tags:
            buckets[tag].append(item)
    for tag, group in sorted(buckets.items()):
        by_phenomenon[tag] = {
            "n": float(len(group)),
            "exact": _mean([item.exact for item in group]),
            "codeswitch_hit": _mean([item.codeswitch_hit for item in group]),
            "family": _mean([item.family for item in group]),
        }
    confusion_counter: Counter[tuple[str, str]] = Counter()
    for item in items:
        if not item.exact:
            confusion_counter[(item.gold, item.predicted)] += 1
    confusion = [(gold, pred, count) for (gold, pred), count in confusion_counter.most_common(20)]
    return RunMetrics(
        model=card.name,
        n=len(items),
        exact=_mean([item.exact for item in items]),
        codeswitch_hit=_mean([item.codeswitch_hit for item in codeswitch_items]),
        family=_mean([item.family for item in items]),
        covered_exact=_mean([item.exact for item in covered]),
        n_covered=len(covered),
        latency_ms_per_example=latency_ms_per_example,
        size_bytes=card.size_bytes,
        by_phenomenon=by_phenomenon,
        confusion=confusion,
        items=items,
    )
