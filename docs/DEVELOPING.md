---
status: draft
audience: public
thesis: interesting-errors-are-linguistic
---

# Developer guide

This is the document to sit with if you want to operate the lab without an
LLM. The README is the public claim. This file is the map of the machinery.
[docs/PHENOMENA.md](PHENOMENA.md) is why the items exist.
[docs/PROJECT_LOG.md](PROJECT_LOG.md) is where a session stopped.

How-tos for the three jobs you will actually do:

- [Run heavier models](how-to/run-heavier-models.md)
- [Add a challenge item](how-to/add-a-challenge-item.md)
- [Add a model](how-to/add-a-model.md)

## Study path

Read in this order. After each step, open the cited file and walk one
example through it on paper. Do not skip the paper step. That is how the
label mapping stops being magic.

1. [docs/PHENOMENA.md](PHENOMENA.md) — what the set is for.
2. `data/challenge.jsonl` — pick `cs-jaen-01`, `short-02`, `rom-01`, `dia-yue-01`, `low-yo-01`. Read the `note` on each.
3. `src/lidlab/schema.py` — `Example`, `Prediction`, `ModelCard`.
4. `src/lidlab/labels.py` — `parse_gold`, `normalize_language`, `normalize_prediction`, `FAMILIES`.
5. `src/lidlab/eval/metrics.py` — `score_item`, then `score_run`.
6. `src/lidlab/models/base.py` and `src/lidlab/models/tfidf.py` — the contract every adapter must keep.
7. `src/lidlab/eval/runner.py` and `src/lidlab/cli.py` — how a run becomes a report.
8. `src/lidlab/analysis/failures.py` — how an error becomes a linguistic note.
9. `tests/` — the executable spec. If a change breaks a test, the change is wrong or the spec needs a deliberate update.

Exercise after step 4: on paper, normalize `__label__eng_Latn`, `__label__zho_Hant`,
`yue_Hant`, and `yor_Latn`. Then check yourself in a REPL:

```bash
uv run python -c "from lidlab.labels import normalize_prediction, parse_gold
print(normalize_prediction('__label__eng_Latn'))
print(normalize_prediction('__label__zho_Hant'))
print(parse_gold('ja+en'))
print(parse_gold('de|no|da|sv|nl'))"
```

Exercise after step 5: score `明日のmeetingは3pmで大丈夫？` (gold `ja+en`)
when the model says `ja`, then when it says `en`, then when it says `fr`.
You should get exact+codeswitch, exact+codeswitch, and a miss.

Exercise after a `tfidf` run: open `reports/latest/failures.md` and explain
three misses in your own words before changing any code.

## What happens in one eval

```
challenge.jsonl ──► Example[] ──► model.predict() ──► Prediction[]
seed.jsonl ──► (tfidf / embed only) ──► trained closed-set model
                         │
                         ▼
              normalize raw labels
                         │
                         ▼
              score_item / score_run
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     report.md      failures.md     {model}.json
```

The CLI entry is `lidlab eval`. `run_eval` loads data, builds each named
model, times `predict`, scores, writes a timestamped directory under
`reports/`, and points `reports/latest` at it.

Closed-set models (`tfidf`, `embed`) train on seed at startup of that
model. They never fit on the challenge set. If you add a challenge item
to `seed.jsonl`, you have leaked the test.

## Directory map

| path | job |
| --- | --- |
| `data/challenge.jsonl` | held-out stress set |
| `data/seed.jsonl` | train-only data for closed-set models |
| `src/lidlab/schema.py` | the three dataclasses everything else speaks |
| `src/lidlab/labels.py` | the only place label strings become comparable |
| `src/lidlab/datasets.py` | JSONL load + structural validation |
| `src/lidlab/paths.py` | repo root, `~/.cache/lidlab`, `LIDLAB_DATA`, `LIDLAB_CACHE` |
| `src/lidlab/models/` | one adapter per backend |
| `src/lidlab/eval/` | metrics, runner, markdown report |
| `src/lidlab/analysis/failures.py` | phenomenon blurbs + per-miss notes |
| `src/lidlab/cli.py` | `eval` and `check-data` |
| `tests/` | data integrity, scoring, tfidf smoke |
| `reports/` | generated; gitignored except `.gitkeep` |

## Data contract

One JSON object per line. Required fields: `id`, `text`, `gold`.

Optional fields:

| field | meaning |
| --- | --- |
| `phenomena` | tags for stratified scoring (`code_switching`, `romanization`, …) |
| `script` | ISO 15924-ish (`Jpan`, `Hans`, `Latn`). Analysis only. Not a gold class. |
| `region` | `BR`, `TW`, `PT`, … Analysis only. |
| `matrix` | the grammatical frame of a mixed utterance (`ja` in `ja+en`) |
| `note` | why the item is hard; copied into `failures.md` on a miss |
| `split` | `challenge` or `seed` |

`gold` notation:

| gold | meaning | exact if prediction is |
| --- | --- | --- |
| `ja` | one language | `ja` |
| `ja+en` | both present | `ja` or `en` |
| `de\|no\|da\|sv\|nl` | any listed label is acceptable | any of those |

Do not mix `+` and `|` on one item. `check-data` and `validate_examples`
reject that. A codeswitch item should also have `matrix` set to one of
the languages in the `+` set.

Phenomenon tags are not mutually exclusive. `short-06` (`hai`) is both
`very_short` and `romanization`. An item can therefore appear in more
than one row of the phenomenon table. That is intended.

Ids should stay stable. Reports and conversations refer to `cs-jaen-01`,
not line numbers.

## Label normalization

Off-the-shelf LID models do not share a label space.

| backend | raw example | after `normalize_prediction` |
| --- | --- | --- |
| fastText lid.176 | `__label__en` | `en`, script `None` |
| GlotLID v3 | `__label__eng_Latn` | `en`, script `Latn` |
| GlotLID | `__label__zho_Hant` | `zh`, script `Hant` |
| XLM-R | `ja` | `ja`, script `None` |

`normalize_language` strips `__label__`, splits `lang_script`, maps
ISO 639-3 → 639-1 when we have a mapping (`jpn` → `ja`, `cmn`/`zho` →
`zh`, `yor` → `yo`), and keeps a short list of 639-3 codes that should
not collapse (`yue`, `haw`, `und`, Egyptian `arz`, …).

If you add a language and scores look insane, you forgot a mapping.
That is the first place to look. Add the ISO3 pair in `ISO3_TO_ISO1`
or the keep-list in `KEEP_ISO3`, then add a test in
`tests/test_labels.py`.

`FAMILIES` is the related-language graph used for the **family** score.
Adding `id`/`ms` style credit for a new pair is a one-line edit there.
Do not put unrelated languages in a family to inflate the number.

## Scoring

All of this lives in `score_item`.

| flag | true when |
| --- | --- |
| `exact` | predicted language ∈ gold `any_of` |
| `codeswitch_hit` | predicted language ∈ gold `all_of` (same set as `any_of` for `+` items) |
| `family` | `exact`, or predicted language shares a family with any acceptable gold |
| `covered` | the model's `supported` set intersects gold `any_of` |

Aggregates in `score_run`:

- **exact / family** — mean over every item
- **codeswitch hit** — mean only over items whose gold contains `+`
- **covered exact** — mean exact over items with `covered=True`
- **n covered** — how many items the model could name at all

A 20-class XLM-R head that marks Yoruba as French is a coverage miss.
`exact` will look bad. `covered exact` tells you whether it is any good
on the languages it actually has. Both numbers belong in the report.
That split is the CommonLID all / cov. idea, applied to this small set.

`tfidf` and `embed` advertise only the seed languages as `supported`.
`fasttext` uses a fixed 176-language list. `xlmr` uses its 20-class
head. `glotlid` derives `supported` from the loaded model's labels.

Confidence is recorded and not used for the official scores. Do not
start thresholding until you can explain a miss without it.

## Model adapters

Every backend implements `LidModel`:

- `card` → `ModelCard(name, kind, supported, size_bytes?, notes)`
- `predict_one(text) → Prediction(language, confidence, raw_label, script?)`
- `predict(texts)` defaults to a loop over `predict_one`

`language` on `Prediction` must already be normalized. Adapters call
`normalize_prediction` (or `normalize_language` for seed-trained heads)
before they return.

| name | extra | trains? | typical weight size | device |
| --- | --- | --- | --- | --- |
| `tfidf` | none | yes, on seed | none | CPU |
| `fasttext` | `fasttext` | no | ~126 MB | CPU |
| `glotlid` | `fasttext` | no | ~1.7 GB | CPU |
| `xlmr` | `neural` | no (pretrained head) | ~1.1 GB | CPU or GPU |
| `embed` | `neural` | yes, on seed | ~470 MB encoder | CPU or GPU |

Install extras from `pyproject.toml`:

```bash
uv sync --extra fasttext --extra dev    # fasttext + glotlid
uv sync --extra neural --extra dev      # xlmr + embed
uv sync --extra all --extra dev         # everything
```

Weights cache under `~/.cache/lidlab` unless `LIDLAB_CACHE` is set.

`tfidf` is character 2–5 grams and `LinearSVC`. It is not a serious LID
system. It is the control that shows the challenge set is doing work:
perfect on native-script controls, empty on romanization, nearly empty
on uncovered languages.

## CLI

```bash
uv run lidlab check-data
uv run lidlab eval --models tfidf
uv run lidlab eval --models fasttext --limit 5
uv run pytest
```

`--limit N` evaluates the first N challenge rows. Use it to prove a
download loaded before you spend a download-and-load cycle on all 94.

Environment:

| variable | default | purpose |
| --- | --- | --- |
| `LIDLAB_DATA` | `<repo>/data` | override JSONL location |
| `LIDLAB_CACHE` | `~/.cache/lidlab` | weight cache |

## How to read a report

`reports/latest/report.md` is the scoreboard. `failures.md` is the
argument. `{model}.json` is the raw item scores if you want to grep.

Read `failures.md` by phenomenon, not by model first. Ask:

1. Did the model recover the **matrix** language on a mixed utterance, or the flashiest script?
2. Did it force a single label on a homograph (`ja`, `no`, `da`)?
3. Did romanization lose the language that native script had for free?
4. Is this a coverage miss (`covered=false`) or a real confusion among languages it knows?

If controls are not near-perfect, stop. The install or the mapping is
broken. Do not narrate stress-case errors on top of a broken baseline.

## Tests as spec

| test | what it locks |
| --- | --- |
| `test_labels.py` | mapping and gold parsing |
| `test_challenge.py` | JSONL validity, required phenomena, seed/challenge split, matrix on `+` items |
| `test_metrics.py` | exact / family / codeswitch / coverage, and the codeswitch aggregate |
| `test_tfidf.py` | English and Japanese controls, and that some challenge items are uncovered |

After any data edit: `uv run lidlab check-data && uv run pytest`.

## Rules that keep the lab honest

1. Seed and challenge never mix.
2. Do not invent a gold class the item does not justify. Isolated `ja` is not Japanese.
3. Do not collapse Cantonese to `zh` in gold just because most models will.
4. Low-resource rows are coverage probes. The `note` should say so when you are not a native annotator.
5. A new model that does not go through `normalize_prediction` is incomparable.
6. Do not chase a higher `exact` by deleting hard items.

## Glossary

| term | meaning here |
| --- | --- |
| matrix language | the grammatical frame of a mixed utterance |
| codeswitch hit | predicted one of the languages that are actually present |
| covered | the model has a name for at least one acceptable gold language |
| family | documented related-language pair, not a language-family tree of the world |
| CommonLID | Mozilla's 2026 web-text LID benchmark; this repo does not reproduce it |
| GlotLID | fastText model with 2000+ language-script labels |
