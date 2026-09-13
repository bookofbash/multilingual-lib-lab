# multilingual-lib-lab — project log

Append-only session history (newest first). The README is the public claim.
This file is where a night of work stopped.

---

## 2026-09-12 — Codeswitch pair (top-2)

**Goal:** Score mixed gold as pair recovery without training a `ja+en` class on the challenge set.

**Done:**
- Adapters return top-2 in `Prediction.extras["alternatives"]`.
- New aggregate **codeswitch pair**: both gold languages ⊆ top-2, mean only over `+` items. `exact` still treats a single `ja` as a hit on `ja+en` (existing spec).
- `tfidf` pair=0.050 (1/20 mixed items: `cs-jaen-07`). Hit stays 0.750. CLI/report column added. Tests 26 passed.

**Open / next:**
1. Re-eval `fasttext`, `glotlid`, `xlmr`, `xlmrft` for the pair column (do not invent those numbers).
2. Optional: same adapters on CommonLID.

---

## 2026-09-12 — XLM-R fine-tune scored on the challenge set

**Goal:** Train `xlmrft` on WiLI (challenge held out) and put it next to papluca `xlmr`.

**Done:**
- Colab T4: smoke `--limit 32`, then `--max-per-lang 400 --epochs 2`. Checkpoint `/root/.cache/lidlab/xlmrft`.
- Eval `reports/20260912T221801Z/`: `xlmr` 0.532 exact / 0.769 covered (65/94); `xlmrft` 0.777 exact / 0.793 covered (92/94). Codeswitch hit 1.0 on both.
- Phenomenon: `related_language` 0.312 → 0.938, `low_resource` 0.077 → 0.846, control 1.0. `very_short` stuck at 0.333. Romanization 0.182 → 0.091. Confusions dump to `yo`.
- Hawaiian uncovered (not in WiLI). Held-out drops: 0. README phenomenon table updated.

**Open / next:**
1. Done: trainer committed (`fa33a34`) and pushed.
2. Codeswitch pair column is in the following log entry.
3. Optional: same adapters on CommonLID.

---

## 2026-09-12 — Start the XLM-R fine-tune

**Goal:** Make the Colab GPU job real in this repo: public LID data, challenge set held out, new `xlmrft` adapter.

**Done:**
- `lidlab train-xlmr` fine-tunes `xlm-roberta-base` on WiLI-2018 (`MartinThoma/wili_2018`), filtered to challenge-relevant languages, capped per language.
- Held-out filter drops any train row whose text matches a challenge item. `--dataset` that names `challenge.jsonl` is refused.
- Adapter `xlmrft` loads `~/.cache/lidlab/xlmrft` (or `LIDLAB_XLMRFT`). `xlmr` stays the 20-class papluca baseline.
- Label mapping: `zh-yue` → `yue` (not `zh`). Recipe: [how-to/finetune-xlmr.md](how-to/finetune-xlmr.md).
- Tests cover the filter and the mapping. The trainer itself was not run (no GPU on the N100).

**Open / next:**
1. On Colab GPU: `pip install -e '.[train,dev]'` then `python -m lidlab.cli train-xlmr --limit 32 --epochs 1`, then the full `--max-per-lang 400 --epochs 2`. Eval `xlmr,xlmrft`. Copy the row into the README.
2. Codeswitch head, or top-2 scoring against `+` labels.
3. Optional: same adapters on CommonLID.

---

## 2026-09-12 — GlotLID on the challenge set

**Goal:** Run `glotlid` alone on the N100 and put the coverage leader on the scoreboard.

**Done:**
- Committed the session log (`90b8c3e`). Local `main` is one commit ahead of origin; not pushed.
- Smoke: `uv run lidlab eval --models glotlid --limit 5` (download ~2 min, then 5/5).
- Full set: `reports/20260912T172307Z/` — 94 items, ~12 s after weights were cached.

**Comparison (94 items):**

| model | exact | codeswitch hit | family | covered exact | n covered |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.511 | 0.750 | 0.553 | 0.649 | 74 / 94 |
| fasttext | 0.755 | 1.000 | 0.809 | 0.772 | 92 / 94 |
| glotlid | 0.809 | 0.950 | 0.819 | 0.809 | 94 / 94 |
| xlmr | 0.532 | 1.000 | 0.553 | 0.769 | 65 / 94 |

GlotLID names every item. The interesting miss is not coverage: `very_short` 0.417 vs fastText 0.667 (`no` → Bribri, `ok` → Luo). One codeswitch miss: `cs-jaen-05` `今日のstandup長すぎた。` → `zh`. Romanization 0.091. `low_resource` and `related_language` are 1.000.

**Open / next:**
1. Run `lidlab train-xlmr` on Colab (see the 2026-09-12 fine-tune entry).
2. Codeswitch head, or top-2 scoring against `+` labels.
3. Optional: same adapters on CommonLID.

---

## 2026-09-12 — Stop for the night

**Goal:** Leave the lab resumable: documented, compared, and on private GitHub.

**Done:**
- Built the compact LID lab: 94-item held-out challenge set, seed-only closed-set training, adapters (`tfidf`, `fasttext`, `glotlid`, `xlmr`, `embed`), coverage-aware scoring, tests.
- Developer docs: `docs/DEVELOPING.md` plus how-tos for running heavier models, adding a challenge item, and adding a model.
- Ran `tfidf`, `fasttext`, and `xlmr` on the N100 (16 GB, no GPU, no Colab). Numbers are in the README comparison table.
- Thesis `interesting-errors-are-linguistic` logged in studio-garden.
- Private GitHub: https://github.com/bookofbash/multilingual-lib-lab (`main`). Origin is HTTPS; this machine has no GitHub SSH key.

**Not on GitHub:** `.venv/`, weight caches (`~/.cache/lidlab`), and generated `reports/` (gitignored except `.gitkeep`). The README table is the committed scoreboard.

**Comparison (94 items):**

| model | exact | codeswitch hit | family | covered exact | n covered |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.511 | 0.750 | 0.553 | 0.649 | 74 / 94 |
| fasttext | 0.755 | 1.000 | 0.809 | 0.772 | 92 / 94 |
| xlmr | 0.532 | 1.000 | 0.553 | 0.769 | 65 / 94 |

Romanization is still empty. Codeswitch hit is 1.0 for both off-the-shelf models. XLM-R's Korean control miss is coverage (`ko` is not in the 20-class head).

**Open / next:**
1. Run `glotlid` (coverage leader, ~1.7 GB, CPU fastText). Local: alone, not with `xlmr`. Colab: CPU/high-RAM runtime, not a GPU; clone with a GitHub token or upload a zip without `.venv`; `python -m lidlab.cli eval --models glotlid --limit 5` then the full set; download `reports/` back and copy the row into the README. Do not paste gold into notebook cells.
2. Fine-tune XLM-R on a larger public LID set; keep this challenge set held out. That is the job that actually wants a Colab GPU.
3. Codeswitch head, or top-2 scoring against `+` labels.
4. Optional: same adapters on CommonLID.

Resume from this file, then [how-to/run-heavier-models.md](how-to/run-heavier-models.md).
