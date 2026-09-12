---
status: draft
audience: public
thesis: interesting-errors-are-linguistic
---

# Add a challenge item

The challenge set is the argument. A new row has to earn its place with a
note about why it is hard. Do not add items to raise `exact`. Do not put a
challenge sentence in `data/seed.jsonl`.

## 1. Decide the gold before the text

Open [../PHENOMENA.md](../PHENOMENA.md). Name the failure mode. Then pick
one gold form:

| gold | meaning | exact if the model says |
| --- | --- | --- |
| `ja` | one language | `ja` |
| `ja+en` | both languages are present | `ja` or `en` |
| `de\|no\|da\|sv\|nl` | genuine ambiguity | any of those |

Do not mix `+` and `|` on one item. `check-data` rejects that.

If the item is mixed, set `matrix` to the grammatical frame (`ja` in
`明日のmeetingは3pmで大丈夫？`). `matrix` must be one of the languages in
the `+` set.

Isolated `ja` is not Japanese. `アルバイト` is Japanese even if the
etymology is German. Do not invent a gold class the text does not justify.
Do not collapse Cantonese to `zh` because most models will.

Low-resource rows are coverage probes unless you are a native annotator.
Say so in `note`.

## 2. Write one JSON line

Append to `data/challenge.jsonl`. Ids are stable; reports refer to them.
Use a prefix you can grep: `cs-`, `rom-`, `short-`, `low-`, `rel-`, `dia-`.

Required fields: `id`, `text`, `gold`.

Useful optional fields:

| field | use |
| --- | --- |
| `phenomena` | tags for the stratified table (`code_switching`, `romanization`, …) |
| `script` | ISO 15924-ish (`Jpan`, `Hans`, `Latn`). Analysis only |
| `region` | `BR`, `TW`, `PT`, … Analysis only |
| `matrix` | required on `+` items |
| `note` | why it is hard; copied into `failures.md` on a miss |
| `split` | omit, or `challenge` |

Tags are not exclusive. `short-06` (`hai`) is `very_short` and
`romanization`. The item will appear in both table rows. That is intended.

Keep `note` specific. "This might be hard" is not a note. "Romanized
Japanese; the Latin script prior points at English or Indonesian" is.

## 3. Walk the gold through the mapping

On paper, then in a REPL:

```bash
uv run python -c "from lidlab.labels import parse_gold
print(parse_gold('YOUR_GOLD'))"
```

If you used a 639-3 code (`yor`, `jpn`) and expected `yo` / `ja`, check
`ISO3_TO_ISO1` in `src/lidlab/labels.py`. If scores later look insane, you
forgot a mapping. Add the pair there, or add the code to `KEEP_ISO3` when
you must not collapse it (`yue`, `haw`). Then add a line to
`tests/test_labels.py`.

## 4. Validate

```bash
uv run lidlab check-data
uv run pytest
```

`tests/test_challenge.py` locks: unique ids, no `+`/`|` in seed, `matrix`
on every mixed item, required phenomenon tags still present, seed still
balanced enough to train. If a test fails, the change is wrong or the spec
needs a deliberate update. Do not weaken a test to sneak a row in.

## 5. Re-run the control before narrating

```bash
uv run lidlab eval --models tfidf
```

Open `reports/latest/failures.md`. If the new item is a miss, the `note`
you wrote should already explain it. If the miss is boring (empty text,
wrong gold, leaked seed sentence), fix the row. If controls dropped below
perfect, stop; the install or the mapping broke.

A closed-set model that has never seen this language should fail coverage.
That is success for a `low_resource` probe. Do not "fix" it by adding the
sentence to `seed.jsonl`.
