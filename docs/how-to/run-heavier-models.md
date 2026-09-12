---
status: draft
audience: public
thesis: interesting-errors-are-linguistic
---

# Run heavier models

`tfidf` is the control. It ships with the repo and needs no weights. The other
four adapters download checkpoints on first load. You do not need Colab for
`fasttext` or `xlmr` on a 16 GB CPU laptop. Colab is a speed option, not a
requirement. Skip `glotlid` until you have an evening and ~2 GB of cache.

## What each extra buys

| name | extra | first download | device | when to run it |
| --- | --- | ---: | --- | --- |
| `tfidf` | none | none | CPU | always; the control |
| `fasttext` | `fasttext` | ~126 MB (`lid.176.bin`) | CPU | first comparison |
| `xlmr` | `neural` | ~1.1 GB | CPU (slow) or GPU | first comparison |
| `embed` | `neural` | ~470 MB | CPU or GPU | after `xlmr` |
| `glotlid` | `fasttext` | ~1.7 GB | CPU | coverage leader; last |

Weights land in `~/.cache/lidlab` unless `LIDLAB_CACHE` is set. Hugging Face
checkpoints go under that cache in `hf/`.

## Install

From the repo root:

```bash
# comparison pair used in the README table
uv sync --extra fasttext --extra neural --extra dev

# everything, including GlotLID
uv sync --extra all --extra dev
```

`neural` pulls PyTorch. On a machine with no NVIDIA GPU that is expected.
Do not install a CUDA wheel to "make it faster" if there is no GPU.

Disk: leave a few GB free. `xlmr` plus `fasttext` is about 1.3 GB. Adding
`glotlid` and `embed` pushes past 3 GB.

## Prove the download before the full set

`--limit 5` evaluates the first five challenge rows. Use it once per new
backend so a bad install fails in a minute instead of after ninety inferences.

```bash
uv run lidlab eval --models fasttext --limit 5
uv run lidlab eval --models xlmr --limit 5
```

If those print `exact=` lines and `wrote reports/.../report.md`, the weights
loaded. Then run the full 94:

```bash
uv run lidlab eval --models tfidf,fasttext,xlmr
```

Reports go to `reports/<timestamp>/` (`report.md`, `failures.md`, per-model
JSON). `reports/latest` is a symlink to the newest run.

On an Intel N100 with 16 GB RAM and no GPU, expect:

- `fasttext` full set: seconds after the one-time download
- `xlmr` full set: several minutes on CPU, not hours
- RAM: stay under 16 GB if you close other heavy processes; do not run
  `glotlid` and `xlmr` at the same time on this machine

If the process is killed, it was almost certainly RAM. Run one model at a
time, or drop `--models` to a single name.

## How to read the new columns

`tfidf` should still be perfect on `control` and empty on `romanization`. If
it is not, the eval is broken; ignore the neural numbers.

Then look at `xlmr` **covered exact** next to **exact**. A 20-class head that
cannot name Yoruba or Hawaiian will look bad on all-items exact. That is a
coverage miss, not a secret about those languages. `fasttext` covers 176
labels, so its all-items and covered numbers should sit closer together.

Read `failures.md` by phenomenon, not by model first. Ask the four questions
in [../DEVELOPING.md](../DEVELOPING.md#how-to-read-a-report).

Copy the overall table into the README when you replace the `tfidf`-only
baseline. Do not round away the coverage split.

## Colab

Use Colab for the XLM-R **fine-tune**, not for scoring `fasttext` or
`xlmr`. Those two already run on the laptop. The challenge set and
scoring live in this repo. A notebook that does not call `lidlab eval`
(or the same `score_run` path) is a different experiment. Keep the JSONL
here; do not paste gold labels into a notebook cell.

The fine-tune recipe is [finetune-xlmr.md](finetune-xlmr.md).
