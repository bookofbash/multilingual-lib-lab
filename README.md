# multilingual-lib-lab

A compact **language identification** lab. The artifact is not another
99%-on-Wikipedia detector. It is a small, inspectable evaluation of how LID
systems behave on the cases production multilingual pipelines actually see:
code-switching, very short input, romanization, related-language confusion,
dialect and script variation, and low-resource coverage.

Positioning: applied NLP with a linguistics foundation — evaluation and
multilingual model behavior, not a from-scratch LID trainer.

## Why this exists

Clean LID numbers overstate the problem. Mozilla's [CommonLID](https://commonlid.org/)
benchmark (2026) already showed that web text is harder than FLORES / UDHR /
Bible evaluations, and that the gap between high-resource and low-resource
languages is the real issue. [GlotLID](https://github.com/cisnlp/GlotLID) is
currently the coverage leader there.

This repo does not reimplement CommonLID. It adds a second axis those
benchmarks are not designed to isolate: **linguistic structure**. The challenge
set is short on purpose. Every item has a note about why it should be hard.

```
明日のmeetingは3pmで大丈夫？
我刚刚push了新的commit
ja
arigatou gozaimasu
你喺邊度呀？
```

See [docs/PHENOMENA.md](docs/PHENOMENA.md) for the linguistic rationale.

## Models

| name | what it is | label set |
| --- | --- | --- |
| `tfidf` | character 2–5 grams + LinearSVC, trained on `data/seed.jsonl` | 16 seed languages |
| `fasttext` | Facebook `lid.176.bin` | 176 languages |
| `glotlid` | [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) v3 | 2000+ language-script labels |
| `xlmr` | [`papluca/xlm-roberta-base-language-detection`](https://huggingface.co/papluca/xlm-roberta-base-language-detection) | 20 languages |
| `embed` | `paraphrase-multilingual-MiniLM-L12-v2` + logistic regression on the seed set | 16 seed languages |

`tfidf` always runs. The others need optional extras and a first-time weight download.

## Setup

```bash
uv sync --extra dev
uv run lidlab check-data
uv run pytest
uv run lidlab eval --models tfidf
```

Heavier comparison (large downloads):

```bash
uv sync --extra all --extra dev
uv run lidlab eval --models tfidf,fasttext,glotlid,xlmr,embed
```

Reports land in `reports/<timestamp>/` (`report.md`, `failures.md`, per-model JSON)
with `reports/latest` pointing at the newest run.

To operate the lab without an LLM, start at
[docs/DEVELOPING.md](docs/DEVELOPING.md). How-tos for the three jobs you will
actually do live in [docs/how-to/](docs/how-to/).

## Scoring

Off-the-shelf models do not share a label space. Predictions are normalized
from fastText (`__label__en`), GlotLID (`__label__eng_Latn`), and ISO-1
classifier heads onto one core.

Gold labels use a small notation:

| gold | meaning |
| --- | --- |
| `ja` | Japanese |
| `ja+en` | both languages are present (code-switching) |
| `de\|no\|da\|sv\|nl` | any listed label is acceptable (homograph) |

Reported numbers:

- **exact** — prediction is in the acceptable gold set
- **codeswitch hit** — prediction is one of the languages in a mixed utterance
- **family** — exact, or a documented related-language pair (id/ms, no/da/sv, zh/yue, …)
- **covered exact** — exact score only on items the model can name (CommonLID-style all / cov. split)

A 20-class XLM-R head that confidently labels Yoruba as French is a coverage
failure, not an F1 footnote.

## Comparison

94 challenge items. `tfidf` is the seed-trained control. `fasttext` is
Facebook `lid.176.bin`. `xlmr` is a 20-class XLM-R head.

| model | exact | codeswitch hit | family | covered exact | n covered |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.511 | 0.750 | 0.553 | 0.649 | 74 / 94 |
| fasttext | 0.755 | 1.000 | 0.809 | 0.772 | 92 / 94 |
| xlmr | 0.532 | 1.000 | 0.553 | 0.769 | 65 / 94 |

| phenomenon | n | tfidf | fasttext | xlmr |
| --- | ---: | ---: | ---: | ---: |
| control | 10 | 1.000 | 1.000 | 0.900 |
| code_switching | 17 | 0.706 | 1.000 | 1.000 |
| related_language | 16 | 0.625 | 0.750 | 0.312 |
| very_short | 12 | 0.333 | 0.667 | 0.333 |
| romanization | 11 | 0.000 | 0.000 | 0.182 |
| low_resource | 13 | 0.077 | 0.692 | 0.077 |

fastText leads all-items exact because it can name 176 languages. XLM-R's
all-items number looks like the sklearn baseline until you read **covered
exact** (0.769 on 65 items): the 20-class head is decent on languages it
has. The Korean control miss (`ko` → `ja`) is coverage — Korean is not in
that head — not a broken install. Indonesian, Malay, and the mainland
Scandinavian languages are also missing, which is why `related_language`
looks worse than `tfidf`.

Romanization still barely moves. Codeswitch hit is 1.0 once the model is
not a tiny n-gram SVM. The remaining misses are still linguistic: isolated
`ja`, romanized `arigatou gozaimasu`, Cantonese collapsed to Mandarin.

## Layout

```
data/challenge.jsonl   # held-out linguistic stress set
data/seed.jsonl        # train-only data for closed-set models
docs/PHENOMENA.md      # why each bucket exists
docs/DEVELOPING.md     # map of the machinery
docs/how-to/           # run models / add an item / add a model
src/lidlab/            # adapters, metrics, CLI
reports/               # generated runs
```

Seed and challenge never mix. Closed-set models train on seed only.

## What this is not

- Not a CommonLID reproduction. Use [mozilla language-id](https://github.com/Mozilla-Data-Collective/language-id) for web-scale numbers.
- Not a claim of native-speaker annotation for every low-resource item. Those rows are coverage probes with documented caveats.
- Not a reason to skip a real multilingual modeling role's PyTorch work. It is evidence that the evaluation and linguistics side of that work is already concrete.

## Next

1. Run `glotlid` and put the coverage leader next to `fasttext`.
2. Fine-tune XLM-R on a larger public LID set and keep this challenge set held out.
3. Add a proper codeswitch head, or evaluate top-2 predictions against `+` labels.
4. Optional: score the same adapters on CommonLID so the stress set and the web benchmark sit side by side.
