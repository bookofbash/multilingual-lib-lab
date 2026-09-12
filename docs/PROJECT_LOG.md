# multilingual-lib-lab — project log

Append-only session history (newest first). The README is the public claim.
This file is where a night of work stopped.

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
