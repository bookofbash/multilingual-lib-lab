from lidlab.datasets import load_challenge, load_seed
from lidlab.eval.metrics import score_run
from lidlab.models.tfidf import TfidfLid


def test_tfidf_recovers_clean_japanese_and_english_controls():
    model = TfidfLid.train(load_seed())
    by_id = {item.id: item for item in load_challenge()}
    english = by_id["ctrl-en-01"]
    japanese = by_id["ctrl-ja-01"]
    assert model.predict_one(english.text).language == "en"
    assert model.predict_one(japanese.text).language == "ja"


def test_tfidf_scores_the_full_challenge_set():
    examples = load_challenge()
    model = TfidfLid.train(load_seed())
    predictions = model.predict([item.text for item in examples])
    metrics = score_run(examples, predictions, model.card, latency_ms_per_example=0.0)
    assert metrics.n == len(examples)
    assert 0.0 <= metrics.exact <= 1.0
    assert metrics.n_covered < metrics.n
