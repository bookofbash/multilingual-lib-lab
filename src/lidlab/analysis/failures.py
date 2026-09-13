from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from lidlab.eval.metrics import RunMetrics
from lidlab.schema import Example

PHENOMENON_BLURBS = {
    "control": (
        "If a model misses clean controls, later stress-case errors are not diagnostic."
    ),
    "code_switching": (
        "Single-label LID cannot represent a mixed utterance. codeswitch hit is recovering "
        "one language; codeswitch pair is recovering both in the top-2. The interesting "
        "question is still whether top-1 is the matrix language or the flashiest script."
    ),
    "very_short": (
        "Function words and interjections starve n-gram and transformer models of context. "
        "Homographs (`ja`, `no`, `da`) make a forced single label the wrong ontology."
    ),
    "homograph": (
        "These items are not model bugs so much as label-set bugs. A serious system should "
        "return a distribution or `und`, not a confident German on isolated `ja`."
    ),
    "romanization": (
        "Latin-script Japanese, Chinese, Korean, Arabic, and Hindi look like each other "
        "and like English. Script priors that help on native orthography become a liability."
    ),
    "related_language": (
        "Indonesian/Malay, Spanish/Portuguese, and the mainland Scandinavian languages "
        "share n-grams. Family credit is recorded separately so a near-miss is visible."
    ),
    "dialect_regional": (
        "Cantonese vs Mandarin and Egyptian vs MSA are often collapsed into a macrolanguage. "
        "That is a product decision, not just an accuracy miss."
    ),
    "script_variant": (
        "Traditional vs Simplified Chinese is the same language under a coarse ISO-1 mapping. "
        "GlotLID can split `zho_Hant` / `zho_Hans`; others generally cannot."
    ),
    "low_resource": (
        "Coverage is the first failure. A 20-class XLM-R head will look confident and still "
        "be wrong. Report all-items and covered-items scores separately."
    ),
    "mixed_script": (
        "A kanji named entity inside an English sentence is enough to hijack script-heavy models."
    ),
    "loanwords": (
        "Lexical borrowing is not code-switching. `アルバイト` is still Japanese; `soirée` is still English."
    ),
    "named_entity": (
        "Proper nouns should not dominate the language decision when the grammatical frame is clear."
    ),
}


def write_failure_notes(path: Path, examples: list[Example], metrics: list[RunMetrics]) -> None:
    by_id = {example.id: example for example in examples}
    lines = [
        "# Failure notes",
        "",
        "These are linguistic readings of errors, not just missed labels.",
        "",
    ]
    for run in metrics:
        lines.append(f"## {run.model}")
        lines.append("")
        misses = [item for item in run.items if not item.exact]
        if not misses:
            lines.append("No exact-match errors on this run.")
            lines.append("")
            continue
        grouped: dict[str, list] = defaultdict(list)
        for item in misses:
            key = item.phenomena[0] if item.phenomena else "untagged"
            grouped[key].append(item)
        for phenomenon, group in grouped.items():
            blurb = PHENOMENON_BLURBS.get(phenomenon, "")
            lines.append(f"### {phenomenon} ({len(group)})")
            lines.append("")
            if blurb:
                lines.append(blurb)
                lines.append("")
            for item in group[:8]:
                example = by_id[item.example_id]
                preview = example.text.replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                lines.append(f"- `{item.example_id}` gold `{item.gold}` → `{item.predicted}` ({item.raw_label})")
                lines.append(f"  - text: {preview}")
                if item.note:
                    lines.append(f"  - why this is hard: {item.note}")
                if example.matrix:
                    lines.append(f"  - matrix language: `{example.matrix}`")
            if len(group) > 8:
                lines.append(f"- … {len(group) - 8} more")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
