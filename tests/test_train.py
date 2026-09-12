from lidlab.cli import main
from lidlab.datasets import load_challenge
from lidlab.labels import normalize_language
from lidlab.models.xlmr_ft import XlmrFtLid
from lidlab.train.heldout import challenge_text_index, filter_training_rows, normalize_text
from lidlab.train.xlmr import TARGET_LANGUAGES, _select_training_kwargs, prepare_rows


def test_challenge_text_is_dropped_from_training_rows():
    challenge = load_challenge()
    item = challenge[0]
    blocked = challenge_text_index(challenge)
    kept, dropped = filter_training_rows(
        [
            (item.text, "en"),
            ("A Wikipedia paragraph about railways that is not in the challenge set.", "en"),
        ],
        blocked,
    )
    assert dropped == 1
    assert kept == [("A Wikipedia paragraph about railways that is not in the challenge set.", "en")]


def test_heldout_index_ignores_whitespace_and_case():
    challenge = load_challenge()
    item = next(row for row in challenge if row.id == "ctrl-en-01")
    blocked = challenge_text_index(challenge)
    padded = f"  {item.text.upper()}  "
    assert normalize_text(padded) in blocked
    kept, dropped = filter_training_rows([(padded, "en")], blocked)
    assert dropped == 1
    assert kept == []


def test_prepare_rows_respects_limit_and_never_keeps_challenge():
    challenge = load_challenge()
    blocked = challenge_text_index(challenge)
    rows = [(item.text, "en") for item in challenge[:3]]
    rows.append(("Unrelated public LID sentence used only in this test.", "en"))
    rows.append(("Another unrelated public LID sentence used only in this test.", "ja"))
    kept, dropped = prepare_rows(rows, blocked=blocked, max_per_lang=8, seed=0, limit=10)
    assert dropped == 3
    assert kept
    challenge_texts = {item.text for item in challenge}
    assert all(text not in challenge_texts for text, _ in kept)


def test_wili_style_labels_map_onto_the_lab_core():
    assert normalize_language("zh-yue") == "yue"
    assert normalize_language("jpn") == "ja"
    assert normalize_language("kor") == "ko"
    assert normalize_language("nob") == "no"
    assert normalize_language("ind") == "id"
    assert normalize_language("may") == "ms"
    assert normalize_language("yor") == "yo"
    assert normalize_language("haw") == "haw"


def test_target_languages_cover_the_xlmr_gaps():
    assert {"ko", "id", "ms", "no", "da", "sv", "yo", "haw", "yue"} <= TARGET_LANGUAGES
    assert {"en", "ja", "zh", "ar", "hi"} <= TARGET_LANGUAGES


def test_train_xlmr_help_does_not_load_the_trainer():
    try:
        main(["train-xlmr", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse to exit after --help")


def test_trainer_refuses_challenge_jsonl_as_source():
    from lidlab.train.xlmr import run_train

    try:
        run_train(dataset_id="data/challenge.jsonl")
    except ValueError as exc:
        assert "challenge.jsonl" in str(exc)
    else:
        raise AssertionError("expected a refusal to train on the challenge set")


def test_training_kwargs_map_warmup_for_transformers_v5():
    v5 = {"warmup_steps", "eval_strategy", "learning_rate", "output_dir"}
    selected = _select_training_kwargs(
        v5,
        {"warmup_ratio": 0.06, "eval_strategy": "epoch", "learning_rate": 2e-5, "unknown": 1},
    )
    assert selected == {"warmup_steps": 0.06, "eval_strategy": "epoch", "learning_rate": 2e-5}


def test_training_kwargs_keep_warmup_ratio_on_v4():
    v4 = {"warmup_ratio", "evaluation_strategy", "learning_rate"}
    selected = _select_training_kwargs(
        v4,
        {"warmup_ratio": 0.06, "eval_strategy": "epoch", "learning_rate": 2e-5},
    )
    assert selected["warmup_ratio"] == 0.06
    assert selected["evaluation_strategy"] == "epoch"


def test_xlmrft_without_checkpoint_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setenv("LIDLAB_XLMRFT", str(tmp_path / "missing"))
    try:
        XlmrFtLid.load()
    except FileNotFoundError as exc:
        assert "train-xlmr" in str(exc)
    else:
        raise AssertionError("expected a missing checkpoint to fail")
