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
| `xlmrft` | `xlm-roberta-base` fine-tuned on WiLI-2018 by `lidlab train-xlmr` | challenge languages; held out |
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
[docs/DEVELOPING.md](docs/DEVELOPING.md). How-tos for the jobs you will
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
- **codeswitch pair** — both gold languages appear in the top-2 (pair recovery without a `ja+en` class)
- **family** — exact, or a documented related-language pair (id/ms, no/da/sv, zh/yue, …)
- **covered exact** — exact score only on items the model can name (CommonLID-style all / cov. split)

A 20-class XLM-R head that confidently labels Yoruba as French is a coverage
failure, not an F1 footnote.

## Comparison

94 challenge items. `tfidf` is the seed-trained control. `fasttext` is
Facebook `lid.176.bin`. `glotlid` is cis-lmu/glotlid v3. `xlmr` is a
20-class XLM-R head. `xlmrft` is `xlm-roberta-base` fine-tuned on
WiLI-2018 with this challenge set held out.

| model | exact | codeswitch hit | codeswitch pair | family | covered exact | n covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.511 | 0.750 | 0.050 | 0.553 | 0.649 | 74 / 94 |
| fasttext | 0.755 | 1.000 | — | 0.809 | 0.772 | 92 / 94 |
| glotlid | 0.809 | 0.950 | — | 0.819 | 0.809 | 94 / 94 |
| xlmr | 0.532 | 1.000 | — | 0.553 | 0.769 | 65 / 94 |
| xlmrft | 0.777 | 1.000 | — | 0.777 | 0.793 | 92 / 94 |

| phenomenon | n | tfidf | fasttext | glotlid | xlmr | xlmrft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 10 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 |
| code_switching | 17 | 0.706 | 1.000 | 0.941 | 1.000 | 1.000 |
| related_language | 16 | 0.625 | 0.750 | 1.000 | 0.312 | 0.938 |
| very_short | 12 | 0.333 | 0.667 | 0.417 | 0.333 | 0.333 |
| romanization | 11 | 0.000 | 0.000 | 0.091 | 0.182 | 0.091 |
| low_resource | 13 | 0.077 | 0.692 | 1.000 | 0.077 | 0.846 |

GlotLID leads all-items exact because it can name the whole set (94 / 94).
`xlmrft` and fastText both cover 92 / 94; the two uncovered `xlmrft`
items are Hawaiian (`haw` is not in WiLI). Papluca XLM-R's all-items
number looks like the sklearn baseline until you read **covered exact**
(0.769 on 65 items). Fine-tuning on Wikipedia mostly bought **coverage**
(65 → 92): `related_language` 0.312 → 0.938, `low_resource` 0.077 →
0.846, control 0.900 → 1.000 (Korean). Covered exact only moved 0.769 →
0.793. The head did not get better at the stress cases Wikipedia does
not contain.

`very_short` is still 0.333. Romanization got **worse** (0.182 → 0.091):
a native-script Wikipedia prior is a liability on `arigatou gozaimasu`.
Hard leftovers dump to Yoruba (`zh`/`ja`/`haw`/`ja` → `yo`), the same
kind of forced label as GlotLID's `no` → Bribri. Codeswitch hit is 1.0
once the model is not a tiny n-gram SVM. **Codeswitch pair** is the
honest next question: both gold languages in the top-2, without a
trained `ja+en` class. `tfidf` is 0.050 (1 / 20 mixed items). The other
adapters now return top-2; fill their pair cells with a re-eval.
GlotLID's one miss is `今日のstandup長すぎた。` snapping to Chinese.

## Layout

```
data/challenge.jsonl   # held-out linguistic stress set
data/seed.jsonl        # train-only data for closed-set models
docs/PHENOMENA.md      # why each bucket exists
docs/DEVELOPING.md     # map of the machinery
docs/PROJECT_LOG.md    # session history; where work last stopped
docs/PROMPT.md         # spend lives in prompt-ops; recipes stay here
docs/how-to/           # run models / fine-tune XLM-R / add an item / add a model
src/lidlab/            # adapters, metrics, CLI, trainer
reports/               # generated runs
```

Seed and challenge never mix. Closed-set models train on seed only.

## What this is not

- Not a CommonLID reproduction. Use [mozilla language-id](https://github.com/Mozilla-Data-Collective/language-id) for web-scale numbers.
- Not a claim of native-speaker annotation for every low-resource item. Those rows are coverage probes with documented caveats.
- Not a reason to skip a real multilingual modeling role's PyTorch work. It is evidence that the evaluation and linguistics side of that work is already concrete.

## Next

1. Optional: score the same adapters on CommonLID so the stress set and the web benchmark sit side by side.
