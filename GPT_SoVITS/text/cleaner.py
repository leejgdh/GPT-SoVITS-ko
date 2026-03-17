from text import cleaned_text_to_sequence
import os

from text import symbols2

_LANGUAGE_MODULE_MAP = {"ja": "japanese", "en": "english", "ko": "korean"}

special = [
    ("￥", "SP2"),
    ("^", "SP3"),
]


def clean_text(text, language, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    symbols = symbols2.symbols

    if language not in _LANGUAGE_MODULE_MAP:
        language = "en"
        text = " "
    for special_s, target_symbol in special:
        if special_s in text:
            return clean_special(text, language, special_s, target_symbol)
    language_module = __import__("text." + _LANGUAGE_MODULE_MAP[language], fromlist=[_LANGUAGE_MODULE_MAP[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
    else:
        phones = language_module.g2p(norm_text)
    word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """특수 침묵 구간 기호(SP) 처리."""
    symbols = symbols2.symbols

    text = text.replace(special_s, ",")
    language_module = __import__("text." + _LANGUAGE_MODULE_MAP[language], fromlist=[_LANGUAGE_MODULE_MAP[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)
