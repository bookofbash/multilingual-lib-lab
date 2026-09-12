from collections import Counter

from lidlab.datasets import load_challenge, load_seed, validate_examples

REQUIRED_PHENOMENA = {
    "control",
    "code_switching",
    "very_short",
    "romanization",
    "related_language",
    "dialect_regional",
    "low_resource",
}


def test_data_files_are_valid():
    challenge = load_challenge()
    seed = load_seed()
    assert validate_examples(challenge) == []
    assert validate_examples(seed) == []
    assert len(challenge) >= 80
    assert len(seed) >= 100


def test_challenge_covers_apple_relevant_phenomena():
    challenge = load_challenge()
    tags = {tag for item in challenge for tag in item.phenomena}
    missing = REQUIRED_PHENOMENA - tags
    assert not missing, missing


def test_seed_is_single_language_only():
    for item in load_seed():
        assert "+" not in item.gold
        assert "|" not in item.gold


def test_codeswitch_items_have_a_matrix_language():
    mixed = [item for item in load_challenge() if "+" in item.gold]
    assert mixed
    for item in mixed:
        assert item.matrix, item.id


def test_seed_is_balanced_enough_to_train():
    counts = Counter(item.gold for item in load_seed())
    assert min(counts.values()) >= 8
    assert {"en", "ja", "zh", "id", "ms", "no", "da", "sv"} <= set(counts)
