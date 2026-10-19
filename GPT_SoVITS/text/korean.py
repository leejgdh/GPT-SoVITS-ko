# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import re
from jamo import h2j, j2hcj
import ko_pron
from g2pk2 import G2p
from loguru import logger

import importlib
import os

# Windows에서 모델 읽기 오류 방지
if os.name == "nt":

    class win_G2p(G2p):
        def check_mecab(self):
            super().check_mecab()
            spam_spec = importlib.util.find_spec("eunjeon")
            non_found = spam_spec is None
            if non_found:
                logger.error("you have to install eunjeon. install it...")
            else:
                installpath = spam_spec.submodule_search_locations[0]
                if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                    import sys
                    from eunjeon import Mecab as _Mecab

                    class Mecab(_Mecab):
                        def get_dicpath(installpath):
                            if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                                import shutil

                                python_dir = os.getcwd()
                                if installpath[: len(python_dir)].upper() == python_dir.upper():
                                    dicpath = os.path.join(os.path.relpath(installpath, python_dir), "data", "mecabrc")
                                else:
                                    if not os.path.exists("TEMP"):
                                        os.mkdir("TEMP")
                                    if not os.path.exists(os.path.join("TEMP", "ko")):
                                        os.mkdir(os.path.join("TEMP", "ko"))
                                    if os.path.exists(os.path.join("TEMP", "ko", "ko_dict")):
                                        shutil.rmtree(os.path.join("TEMP", "ko", "ko_dict"))

                                    shutil.copytree(
                                        os.path.join(installpath, "data"), os.path.join("TEMP", "ko", "ko_dict")
                                    )
                                    dicpath = os.path.join("TEMP", "ko", "ko_dict", "mecabrc")
                            else:
                                dicpath = os.path.abspath(os.path.join(installpath, "data/mecabrc"))
                            return dicpath

                        def __init__(self, dicpath=get_dicpath(installpath)):
                            super().__init__(dicpath=dicpath)

                    sys.modules["eunjeon"].Mecab = Mecab

    G2p = win_G2p


from text.korean_numbers import normalize_numbers
from text.symbols2 import symbols

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
        # ('ㄵ', 'ㄴㅈ'),
        # ('ㄶ', 'ㄴㅎ'),
        # ('ㄺ', 'ㄹㄱ'),
        # ('ㄻ', 'ㄹㅁ'),
        # ('ㄼ', 'ㄹㅂ'),
        # ('ㄽ', 'ㄹㅅ'),
        # ('ㄾ', 'ㄹㅌ'),
        # ('ㄿ', 'ㄹㅍ'),
        # ('ㅀ', 'ㄹㅎ'),
        # ('ㅄ', 'ㅂㅅ'),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("t͡ɕ", "ʧ"),
        ("d͡ʑ", "ʥ"),
        ("ɲ", "n^"),
        ("ɕ", "ʃ"),
        ("ʷ", "w"),
        ("ɭ", "l`"),
        ("ʎ", "ɾ"),
        ("ɣ", "ŋ"),
        ("ɰ", "ɯ"),
        ("ʝ", "j"),
        ("ʌ", "ə"),
        ("ɡ", "g"),
        ("\u031a", "#"),
        ("\u0348", "="),
        ("\u031e", ""),
        ("\u0320", ""),
        ("\u0339", ""),
    ]
]


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
            new_text += text[i : i + 3] + " " + "ㄴ"
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text


def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


def korean_to_lazy_ipa(text):
    text = normalize_numbers(text, _g2p.mecab)
    text = latin_to_hangul(text)
    text = re.sub("[\uac00-\ud7af]+", lambda x: ko_pron.romanise(x.group(0), "ipa").split("] ~ [")[0], text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


_g2p = G2p()


def korean_to_ipa(text):
    text = normalize_numbers(text, _g2p.mecab)
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace("ʧ", "tʃ").replace("ʥ", "dʑ")


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        " ": "空",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "停"
    return ph


def g2p(text):
    text = normalize_numbers(text, _g2p.mecab)
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    # text = "".join([post_replace_ph(i) for i in text])
    text = [post_replace_ph(i) for i in text]
    return text
