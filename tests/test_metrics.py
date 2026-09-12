from lidlab.eval.metrics import score_item, score_run
from lidlab.schema import Example, ModelCard, Prediction


def _example(gold: str, **kwargs) -> Example:
    payload = {"id": "x", "text": "hello", "gold": gold}
    payload.update(kwargs)
    return Example.from_dict(payload)


def test_exact_and_family_and_codeswitch():
    supported = frozenset({"en", "ja", "id", "ms"})
    exact = score_item(_example("ja"), Prediction("ja", 0.9, "ja"), supported)
    assert exact.exact and exact.family

    family = score_item(_example("id"), Prediction("ms", 0.7, "ms"), supported)
    assert not family.exact
    assert family.family

    mixed = score_item(
        _example("ja+en", phenomena=["code_switching"], matrix="ja"),
        Prediction("ja", 0.6, "ja"),
        supported,
    )
    assert mixed.exact
    assert mixed.codeswitch_hit

    snap = score_item(
        _example("en+ja", phenomena=["code_switching"], matrix="en"),
        Prediction("fr", 0.4, "fr"),
        supported,
    )
    assert not snap.exact
    assert not snap.codeswitch_hit


def test_coverage_marks_unsupported_low_resource():
    supported = frozenset({"en", "ja", "zh"})
    item = score_item(_example("yo"), Prediction("en", 0.8, "en"), supported)
    assert not item.covered
    assert not item.exact


def test_codeswitch_aggregate_ignores_monolingual_items():
    examples = [_example("en", id="a"), _example("ja+en", id="b", phenomena=["code_switching"])]
    predictions = [Prediction("en", 1.0, "en"), Prediction("fr", 0.2, "fr")]
    card = ModelCard(name="toy", kind="test", supported=frozenset({"en", "ja"}))
    metrics = score_run(examples, predictions, card, latency_ms_per_example=0.0)
    assert metrics.exact == 0.5
    assert metrics.codeswitch_hit == 0.0
