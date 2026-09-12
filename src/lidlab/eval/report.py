from __future__ import annotations

from pathlib import Path

from lidlab.eval.metrics import RunMetrics
from lidlab.schema import Example


def write_report(path: Path, examples: list[Example], metrics: list[RunMetrics]) -> None:
    lines = [
        "# LID lab report",
        "",
        f"Challenge items: **{len(examples)}**",
        "",
        "Scores are not web-scale accuracy. They measure behavior on a linguistics-designed",
        "stress set: code-switching, short input, romanization, related languages,",
        "dialect/script variation, and low-resource coverage.",
        "",
        "## Overall",
        "",
        "| model | exact | codeswitch hit | family | covered exact | n covered | ms / item | size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        size = _fmt_size(item.size_bytes)
        lines.append(
            "| {model} | {exact:.3f} | {cs:.3f} | {family:.3f} | {cov:.3f} | {n_cov} | {ms:.1f} | {size} |".format(
                model=item.model,
                exact=item.exact,
                cs=item.codeswitch_hit,
                family=item.family,
                cov=item.covered_exact,
                n_cov=item.n_covered,
                ms=item.latency_ms_per_example,
                size=size,
            )
        )
    lines.extend(["", "## By phenomenon", ""])
    phenomena = sorted({tag for item in metrics for tag in item.by_phenomenon})
    header = "| phenomenon | n | " + " | ".join(f"{item.model} exact" for item in metrics) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in metrics) + " |"
    lines.extend([header, sep])
    for tag in phenomena:
        counts = [item.by_phenomenon.get(tag, {}).get("n", 0.0) for item in metrics]
        n = int(max(counts) if counts else 0)
        cells = []
        for item in metrics:
            bucket = item.by_phenomenon.get(tag)
            cells.append(f"{bucket['exact']:.3f}" if bucket else "—")
        lines.append(f"| {tag} | {n} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Top confusions", ""])
    for item in metrics:
        lines.append(f"### {item.model}")
        lines.append("")
        if not item.confusion:
            lines.append("No exact-match errors.")
            lines.append("")
            continue
        lines.append("| gold | predicted | n |")
        lines.append("| --- | --- | ---: |")
        for gold, pred, count in item.confusion[:10]:
            lines.append(f"| {gold} | {pred} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## How to read this",
            "",
            "- **exact**: predicted language is in the acceptable gold set.",
            "- **codeswitch hit**: predicted language is one of the languages in a mixed utterance.",
            "  Off-the-shelf models have no `ja+en` class, so this is the honest partial credit.",
            "- **family**: exact, or a related-language confusion (id/ms, no/da/sv, zh/yue, …).",
            "- **covered exact**: exact score on items the model can name at all.",
            "  This is the CommonLID-style coverage split: do not punish a 20-class XLM-R",
            "  head for missing Yoruba if you also report the all-items number.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    mb = size_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{size_bytes} B"
