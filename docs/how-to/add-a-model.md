---
status: draft
audience: public
thesis: interesting-errors-are-linguistic
---

# Add a model

Every backend is a thin adapter. Scoring, reports, and label mapping stay
outside it. If the new class returns raw `__label__eng_Latn` as `language`,
the comparison is theater.

## 1. Implement `LidModel`

Add `src/lidlab/models/<name>.py`. Copy the shape of
`src/lidlab/models/tfidf.py` (trains on seed) or
`src/lidlab/models/fasttext_lid.py` (loads a checkpoint).

The contract in `src/lidlab/models/base.py`:

- `card` → `ModelCard(name, kind, supported, size_bytes?, notes)`
- `predict_one(text) → Prediction(language, confidence, raw_label, script?)`
- `predict(texts)` already loops; override only if you have a real batch path

`language` must already be normalized. Call
`normalize_prediction` (or `normalize_language` for a seed-trained head)
before you return. Keep the backend's original string in `raw_label`.

`supported` is the set of languages the model can name, after the same
mapping. Closed-set heads (`tfidf`, `embed`) advertise the seed languages.
A 20-class XLM-R head advertises those 20. `covered exact` is computed
from this set. If you leave `supported` empty, everything looks covered.

Weights cache under `cache_dir()` (`~/.cache/lidlab`, or `LIDLAB_CACHE`).
Do not download into the repo.

## 2. Register it

In `src/lidlab/models/__init__.py`:

1. Add the name to `MODEL_KINDS`.
2. Add a branch in `build_model` that imports lazily and calls `load()` or
   `train(seed)`.

Lazy imports keep `uv run lidlab eval --models tfidf` from pulling
PyTorch. Closed-set models train on the seed argument. They never fit on
the challenge set.

If the backend needs a library the core install should not carry, add an
extra in `pyproject.toml` (`fasttext`, `neural`, or a new one). Raise
`ImportError` with the extra name from `load()` / `train()`, the way
`FastTextLid` and `XlmrLid` do.

## 3. Map labels you have not seen before

Work a raw label through `normalize_prediction` in a REPL before the first
eval. FastText looks like `__label__en`. GlotLID looks like
`__label__eng_Latn`. Some heads use `jpn` or `cmn`.

If the mapped language is wrong, edit `ISO3_TO_ISO1` or `KEEP_ISO3` in
`src/lidlab/labels.py` and add a case to `tests/test_labels.py`. Do not
special-case inside the adapter unless the backend uses a private code
that should never leak into the shared map.

Related-language credit lives in `FAMILIES`. Add a pair only when a
near-miss is a documented confusion (id/ms, no/da/sv, zh/yue). Do not add
unrelated languages to inflate **family**.

## 4. Prove it without the full set

```bash
uv run pytest
uv run lidlab eval --models yourname --limit 5
```

`--limit 5` is the first five challenge rows, which include a clean
English control. If that control is not `en`, the mapping or the checkpoint
is wrong. Do not run all 94 until the control is right.

Then run against `tfidf` so you have a baseline in the same report:

```bash
uv run lidlab eval --models tfidf,yourname
```

Read `reports/latest/report.md`. If controls are not near-perfect, stop.
If **covered exact** equals **exact** on a tiny label set, `supported` is
probably wrong.

## 5. What not to do

- Do not score inside the adapter.
- Do not train on `challenge.jsonl`.
- Do not chase a higher `exact` by dropping hard items.
- Do not threshold on `confidence` until you can explain a miss without it.
- Do not add a model whose labels skip `normalize_prediction`.
