"""Normalize the messy LID label space onto a small ISO-like core.

Off-the-shelf models disagree about everything: fastText uses ISO 639-1
(``__label__en``), GlotLID uses ISO 639-3 plus script (``__label__eng_Latn``),
and XLM-R checkpoints often use bare 639-1. Scoring has to happen after
this mapping, or comparisons are theater.
"""

from __future__ import annotations

from dataclasses import dataclass

ISO3_TO_ISO1 = {
    "afr": "af",
    "amh": "am",
    "ara": "ar",
    "arb": "ar",
    "bel": "be",
    "ben": "bn",
    "bos": "bs",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cmn": "zh",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "epo": "eo",
    "est": "et",
    "eus": "eu",
    "fas": "fa",
    "fin": "fi",
    "fra": "fr",
    "gle": "ga",
    "glg": "gl",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "hye": "hy",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "ja",
    "kat": "ka",
    "kor": "ko",
    "lat": "la",
    "lav": "lv",
    "lit": "lt",
    "lvs": "lv",
    "mar": "mr",
    "mkd": "mk",
    "msa": "ms",
    "nld": "nl",
    "nno": "nn",
    "nob": "no",
    "nor": "no",
    "pes": "fa",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "srp": "sr",
    "swe": "sv",
    "swh": "sw",
    "swa": "sw",
    "tam": "ta",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
    "yid": "yi",
    "yor": "yo",
    "zho": "zh",
    "zsm": "ms",
}

# 639-3 codes we keep as-is because they have no 639-1 mapping we want to collapse.
KEEP_ISO3 = {
    "yue",
    "wuu",
    "nan",
    "haw",
    "arz",
    "apc",
    "acm",
    "ajp",
    "ary",
    "und",
    "zxx",
}

FAMILIES: dict[str, frozenset[str]] = {
    "scandinavian": frozenset({"no", "nb", "nn", "da", "sv"}),
    "iberian": frozenset({"es", "pt", "gl", "ca"}),
    "malay": frozenset({"id", "ms"}),
    "sinitic": frozenset({"zh", "yue", "wuu", "nan"}),
    "hindustani": frozenset({"hi", "ur"}),
    "serbo_croatian": frozenset({"hr", "sr", "bs", "cnr"}),
    "west_slavic": frozenset({"cs", "sk"}),
    "east_slavic": frozenset({"ru", "uk", "be"}),
}

SCRIPT_ALIASES = {
    "hans": "Hans",
    "hant": "Hant",
    "latn": "Latn",
    "cyrl": "Cyrl",
    "arab": "Arab",
    "jpan": "Jpan",
    "kore": "Kore",
    "deva": "Deva",
    "thai": "Thai",
    "grek": "Grek",
    "hebr": "Hebr",
    "armn": "Armn",
    "geor": "Geor",
    "ethi": "Ethi",
}


@dataclass(frozen=True)
class GoldSpec:
    """Parsed gold label.

    ``all_of`` is the code-switch set (``+``).
    ``any_of`` is the acceptable-alternative set (``|``).
    A single-language item has both equal to ``{lang}``.
    """

    raw: str
    all_of: frozenset[str]
    any_of: frozenset[str]
    codeswitch: bool
    ambiguous: bool


def parse_gold(raw: str) -> GoldSpec:
    text = raw.strip()
    if "+" in text and "|" in text:
        raise ValueError(f"gold cannot mix + and |: {raw}")
    if "+" in text:
        parts = frozenset(normalize_language(p) for p in text.split("+") if p)
        return GoldSpec(raw=text, all_of=parts, any_of=parts, codeswitch=True, ambiguous=False)
    if "|" in text:
        parts = frozenset(normalize_language(p) for p in text.split("|") if p)
        return GoldSpec(raw=text, all_of=parts, any_of=parts, codeswitch=False, ambiguous=True)
    lang = normalize_language(text)
    return GoldSpec(
        raw=text,
        all_of=frozenset({lang}),
        any_of=frozenset({lang}),
        codeswitch=False,
        ambiguous=False,
    )


def normalize_language(label: str) -> str:
    """Map a model or gold label onto a comparable language code."""
    token = label.strip()
    if token.startswith("__label__"):
        token = token[len("__label__") :]
    token = token.replace(" ", "")
    script = None
    if "_" in token:
        lang_part, script_part = token.split("_", 1)
        token = lang_part
        script = SCRIPT_ALIASES.get(script_part.lower(), script_part)
    if "-" in token:
        token = token.split("-", 1)[0]
    lower = token.lower()
    if lower in {"zh-cn", "zh-tw", "zh-hk"}:
        return "zh"
    if lower in KEEP_ISO3:
        return lower
    if lower in ISO3_TO_ISO1:
        return ISO3_TO_ISO1[lower]
    return lower


def normalize_prediction(label: str) -> tuple[str, str | None]:
    """Return ``(language, script)`` from a raw model label."""
    token = label.strip()
    if token.startswith("__label__"):
        token = token[len("__label__") :]
    script = None
    if "_" in token:
        lang_part, script_part = token.split("_", 1)
        script = SCRIPT_ALIASES.get(script_part.lower(), script_part)
        token = lang_part
    return normalize_language(token), script


def family_of(lang: str) -> str | None:
    for name, members in FAMILIES.items():
        if lang in members:
            return name
    return None


def same_family(a: str, b: str) -> bool:
    left = family_of(a)
    return bool(left and left == family_of(b))


def model_covers(supported: frozenset[str], gold: GoldSpec) -> bool:
    """True if the model can name at least one acceptable language."""
    if not supported:
        return True
    return bool(gold.any_of & supported)
