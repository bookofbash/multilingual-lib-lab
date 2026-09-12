---
status: draft
audience: public
thesis: interesting-errors-are-linguistic
---

# Fine-tune XLM-R

`xlmr` is papluca's 20-class head. It cannot name Korean, Indonesian,
Yoruba, or Hawaiian. This job trains a **new** head, `xlmrft`, on a
public LID set and evaluates it on the same held-out challenge JSONL.
The point is not a higher Wikipedia number. The point is whether a
wider neural head still fails the linguistic cases.

Do this on a Colab GPU. The N100 can load `xlmr` for eval. It should
not run this trainer.

## What it trains on

[WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018)
Wikipedia paragraphs, filtered to languages the challenge set actually
uses (papluca's 20 plus the coverage gaps: `ko`, `id`, `ms`, `no`,
`da`, `sv`, `yo`, `haw`, `yue`, …). Cap per language so a T4 finishes
in one sitting.

`data/challenge.jsonl` is not a source. The trainer drops any WiLI row
whose text matches a challenge item. Do not pass the challenge file as
`--dataset`. Do not paste gold labels into a notebook cell.

WiLI is clean Wikipedia. That is the distribution the thesis is
suspicious of. Training on it, then scoring the stress set, is the
experiment.

## Install

GPU runtime (**Runtime → Change runtime type → T4 GPU**). Upload
`multilingual-lib-lab-colab.zip` (no `.venv`) with the Colab file
sidebar, or clone the private repo with a GitHub token.

Colab cells are Python. Shell lines need `!`. Unzip, then `%cd` so the
editable install sees `pyproject.toml`.

```
!unzip -q /content/multilingual-lib-lab-colab.zip -d /content
%cd /content/multilingual-lib-lab
%pip install -e '.[train,dev]'
!python -m lidlab.cli check-data
!python -m lidlab.cli train-xlmr --limit 32 --epochs 1
```

`--limit 32` proves the Hub download, the held-out filter, and one
optimizer step. If that prints `wrote .../xlmrft` and `xlmrft: n=`,
run the real job:

```
!python -m lidlab.cli train-xlmr --max-per-lang 400 --epochs 2
```

If you are in a terminal instead of a notebook, drop the `!` / `%`
prefixes and run the same commands from the repo root.

Colab often asks to **restart the kernel** after `%pip install`. That is
normal. Disk stays (`/content/…`, Hub cache). The notebook forgets
`%cd`. After Restart, do **not** unzip or pip again unless
`/content/multilingual-lib-lab/pyproject.toml` is gone.

```
%cd /content/multilingual-lib-lab
!python -m lidlab.cli check-data
!python -m lidlab.cli train-xlmr --limit 32 --epochs 1
```

If `check-data` says `No module named lidlab`, re-run only:

```
%cd /content/multilingual-lib-lab
%pip install -e '.[train,dev]'
```

then restart once more and go back to `check-data` / `train-xlmr`.
If unzip is missing too, upload the zip again. The 1.1 GB model should
still be in the Hub cache on this VM.

Checkpoint default: `~/.cache/lidlab/xlmrft` (or `$LIDLAB_CACHE/xlmrft`,
or `$LIDLAB_XLMRFT`). Weights stay out of git.

On a T4, expect tens of minutes after the first-time downloads
(`xlm-roberta-base` plus WiLI train). If the runtime disconnects, the
partial `runs/` folder is not the adapter; only a directory with
`config.json` loads.

## Prove the head before the full set

```
!python -m lidlab.cli eval --models xlmr,xlmrft --limit 5
```

The English control must be `en` on both. `xlmrft` should also be able
to name Korean (`ctrl-ko-01` is not in the first five; that is a
coverage check on the full set). Then:

```
!python -m lidlab.cli eval --models xlmr,xlmrft
```

Read `reports/latest/report.md`. Copy **exact** and **covered exact**
into the README next to the 20-class `xlmr` row. Do not round away the
coverage split. If `xlmrft` covered exact does not beat `xlmr` on
languages both can name, the fine-tune is not done.

Download `reports/` back to the laptop. Leave the checkpoint in Colab
cache unless you want to eval locally later: copy the `xlmrft`
directory into `~/.cache/lidlab/xlmrft` or set `LIDLAB_XLMRFT`.

## What not to do

- Do not train on `challenge.jsonl`.
- Do not fine-tune papluca's 20-class checkpoint and call that coverage.
  Start from `xlm-roberta-base` so the head can grow.
- Do not run this trainer next to `glotlid` on a 16 GB CPU machine.
- Do not paste challenge gold into cells. Call `lidlab`.
