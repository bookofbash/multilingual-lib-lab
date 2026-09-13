# Why these items are in the set

This is not a random multilingual sample. Each bucket is a failure mode that
clean LID benchmarks (FLORES, Bible, UDHR, Wikipedia) systematically hide.

Mozilla's CommonLID result is the backdrop: web text is already harder than
those clean sets, and the gap between high-resource and low-resource languages
is the actual product problem. This lab adds a second axis that CommonLID is
not designed to isolate — **linguistic structure**.

## Control

If a model cannot identify a full sentence of English, Japanese, or Mandarin,
nothing else in the report is diagnostic. Controls exist so a broken install
does not get narrated as a linguistics finding.

## Code-switching

`明日のmeetingは3pmで大丈夫？` has a Japanese grammatical frame and English
insertions. `I'll send the 資料 after lunch.` is the opposite. Off-the-shelf
LID has no `ja+en` class, so exact match on a mixed gold label is the wrong
question. The honest scores are:

- did it recover **one** of the languages (`codeswitch hit`)
- did both languages appear in the **top-2** (`codeswitch pair`)
- did it recover the **matrix** language (in the notes)
- did script hijack the decision (kanji / hanzi pulling an English frame to `ja` / `zh`)

Treating intra-sentential switching as noise is how production multilingual
systems silently misroute language-id dependent pipelines.

## Very short input and homographs

`ja`, `no`, and `da` are not just short. They are homographs across languages
and they collide with ISO language codes. Isolated `ja` is a perfectly good
German, Norwegian, Danish, Swedish, or Dutch utterance. It is not Japanese.
A model that returns a single high-confidence label here is answering a
question the data does not ask.

## Romanization

Japanese, Mandarin, Korean, Arabic, and Hindi all have widely used Latin-script
forms. Native-script accuracy is cheap once the writing system is visible.
Romanized `arigatou gozaimasu` and `ni hao ma` are the actual hard case:
the script prior that helped a moment ago now points at English, Indonesian,
or "undetermined."

## Related languages

Indonesian/Malay, Spanish/Portuguese, and mainland Scandinavian languages share
character n-grams and a great deal of lexicon. Family credit is recorded so a
near-miss is visible instead of being flattened into a single accuracy number.
The seed set includes close pairs on purpose, so a closed-set sklearn baseline
has a chance — and so its confusions are interpretable.

## Dialect, region, script

Written Cantonese (`你喺邊度呀？`) is not Mandarin. Egyptian Arabic is not MSA.
Traditional and Simplified Chinese are the same language under ISO 639-1 and
different labels under GlotLID (`zho_Hant` / `zho_Hans`). Collapsing these is
sometimes a product decision. The set makes that decision visible.

## Low-resource coverage

Welsh, Irish, Yoruba, Basque, Hawaiian, and Amharic are here less as a claim
of native-speaker annotation quality and more as a coverage probe. A 20-class
XLM-R head will look confident and still be wrong. That is why the report
splits **all-items exact** from **covered exact**, following the CommonLID
all / cov. distinction.

## Mixed script, loans, named entities

`Tokyo 東京 is beautiful in April` is English. `アルバイト` is Japanese, even
though the etymology is German. `soirée` inside an English sentence is still
English. LID that fires on the most "foreign" span will break every
downstream filter that trusts it.
