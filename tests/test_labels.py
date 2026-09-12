from lidlab.labels import normalize_language, normalize_prediction, parse_gold, same_family


def test_fasttext_and_glotlid_labels_collapse():
    assert normalize_language("__label__en") == "en"
    assert normalize_language("__label__eng_Latn") == "en"
    assert normalize_language("jpn_Jpan") == "ja"
    assert normalize_language("cmn_Hans") == "zh"
    assert normalize_language("yue_Hant") == "yue"
    assert normalize_language("yor_Latn") == "yo"
    assert normalize_language("zh-yue") == "yue"
    assert normalize_language("zh-cn") == "zh"


def test_prediction_keeps_script():
    language, script = normalize_prediction("__label__zho_Hant")
    assert language == "zh"
    assert script == "Hant"


def test_codeswitch_and_ambiguity_parse():
    mixed = parse_gold("ja+en")
    assert mixed.codeswitch
    assert mixed.all_of == frozenset({"ja", "en"})
    ambi = parse_gold("de|no|da|sv|nl")
    assert ambi.ambiguous
    assert "de" in ambi.any_of
    assert parse_gold("zh").any_of == frozenset({"zh"})


def test_related_language_families():
    assert same_family("id", "ms")
    assert same_family("no", "da")
    assert same_family("zh", "yue")
    assert not same_family("ja", "zh")
