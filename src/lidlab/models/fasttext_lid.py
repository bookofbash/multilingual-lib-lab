from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from lidlab.labels import normalize_prediction
from lidlab.models.base import LidModel
from lidlab.paths import cache_dir
from lidlab.schema import ModelCard, Prediction

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_LANGUAGES = frozenset(
    {
        "af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb",
        "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs",
        "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da",
        "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et",
        "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom",
        "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia",
        "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk",
        "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez",
        "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk",
        "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah",
        "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa",
        "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru",
        "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl",
        "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
        "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls",
        "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh",
    }
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class FastTextLid(LidModel):
    """Facebook's 176-language fastText LID."""

    def __init__(self, model, path: Path) -> None:
        self.model = model
        self._card = ModelCard(
            name="fasttext",
            kind="fasttext-lid.176",
            supported=FASTTEXT_LANGUAGES,
            size_bytes=path.stat().st_size if path.exists() else None,
            notes="Official lid.176.bin. Strong on clean long text; weak on short and romanized input.",
        )

    @property
    def card(self) -> ModelCard:
        return self._card

    @classmethod
    def load(cls) -> FastTextLid:
        try:
            import fasttext
        except ImportError as exc:
            raise ImportError("fasttext extra required: pip install 'lidlab[fasttext]'") from exc

        path = _download(FASTTEXT_URL, cache_dir() / "lid.176.bin")
        # fastText warns on isolated newlines; callers should already strip.
        model = fasttext.load_model(str(path))
        return cls(model, path)

    def predict_one(self, text: str) -> Prediction:
        cleaned = text.replace("\n", " ").strip() or " "
        labels, probs = self.model.predict(cleaned, k=1)
        raw = labels[0]
        language, script = normalize_prediction(raw)
        return Prediction(
            language=language,
            confidence=float(probs[0]),
            raw_label=raw,
            script=script,
        )
