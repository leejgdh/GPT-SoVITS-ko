"""ko/ja/en + 숫자 언어 분할.

split_lang 으로 1차 분할 → 카테고리 충돌이 잦은 ja/ko 구간만 정규식으로 재분할 후
동일 언어를 병합한다. 중국어/번체 분기는 제거됐다 — cleaner.py 도 ja/en/ko 만 지원.
"""
import logging
import re
from pathlib import Path

# split_lang 이 jieba 를 transitive 로 사용 — 경고만 무음 처리하고 직접 호출은 없음.
import jieba

jieba.setLogLevel(logging.CRITICAL)

import fast_langdetect
from split_lang import LangSplitter

# fast_langdetect 모델 캐시 위치를 프로젝트 내부로 고정.
_fast_langdetect_cache = Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"
_fast_langdetect_cache.mkdir(parents=True, exist_ok=True)
fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
    fast_langdetect.infer.LangDetectConfig(cache_dir=_fast_langdetect_cache)
)


_FULL_EN_PATTERN = re.compile(
    r"^(?=.*[A-Za-z])[A-Za-z0-9\s -~ -⁯　-〿＀-￯]+$"
)
_JA_PATTERN = re.compile(
    r"([ぁ-ゖ゙゚ァ-ヺー]+"
    r"(?:[0-9、-〜。！？.!?… ]+[ぁ-ゖ゙゚ァ-ヺー]*)*)"
)
_KO_PATTERN = re.compile(
    r"([ᄀ-ᇿ㄰-㆏가-힯]+"
    r"(?:[0-9、-〜。！？.!?… ]+[ᄀ-ᇿ㄰-㆏가-힯]*)*)"
)
_PUNCT = {",", ".", "!", "?", "，", "。", "！", "？"}


def _is_full_en(text: str) -> bool:
    return bool(_FULL_EN_PATTERN.match(text))


def _split_by_pattern(tag_lang: str, item: dict, pattern: re.Pattern) -> list[dict]:
    """item.text 안에서 pattern 매칭 구간을 tag_lang 으로 잘라낸다."""
    lang_list: list[dict] = []
    tag = 0
    for match in pattern.finditer(item["text"]):
        if match.start() > tag:
            lang_list.append({"lang": item["lang"], "text": item["text"][tag : match.start()]})
        tag = match.end()
        lang_list.append({"lang": tag_lang, "text": item["text"][match.start() : match.end()]})
    if tag < len(item["text"]):
        lang_list.append({"lang": item["lang"], "text": item["text"][tag:]})
    return lang_list


def _merge_lang(lang_list: list[dict], item: dict) -> list[dict]:
    if lang_list and item["lang"] == lang_list[-1]["lang"]:
        lang_list[-1]["text"] += item["text"]
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter:
    # GSV 가 지원하는 언어 = ko / ja / en (+ digit).
    DEFAULT_LANG_MAP = {
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    @staticmethod
    def getTexts(text: str, default_lang: str = "") -> list[dict]:
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        lang_splitter.merge_across_digit = False
        substr = lang_splitter.split_by_lang(text=text)

        lang_list: list[dict] = []
        have_num = False

        for item in substr:
            dict_item = {"lang": item.lang, "text": item.text}

            if dict_item["lang"] == "digit":
                if default_lang:
                    dict_item["lang"] = default_lang
                else:
                    have_num = True
                lang_list = _merge_lang(lang_list, dict_item)
                continue

            # 짧은 영문이 다른 언어로 잘못 인식되는 경우 보정.
            if _is_full_en(dict_item["text"]):
                dict_item["lang"] = "en"
                lang_list = _merge_lang(lang_list, dict_item)
                continue

            if default_lang:
                dict_item["lang"] = default_lang
                lang_list = _merge_lang(lang_list, dict_item)
                continue

            # default_lang 미지정: 일본어/한국어가 다른 카테고리로 잡힌 경우 재분할.
            ja_list = _split_by_pattern("ja", dict_item, _JA_PATTERN) if dict_item["lang"] != "ja" else []
            if not ja_list:
                ja_list = [dict_item]

            temp_list: list[dict] = []
            for ko_item in ja_list:
                ko_list = _split_by_pattern("ko", ko_item, _KO_PATTERN) if ko_item["lang"] != "ko" else []
                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            for temp_item in temp_list:
                # 미식별 카테고리는 직전 lang 으로 흡수 (없으면 en).
                if temp_item["lang"] == "x":
                    temp_item["lang"] = lang_list[-1]["lang"] if lang_list else "en"
                lang_list = _merge_lang(lang_list, temp_item)

        # default_lang 가 없을 때만 'digit' 가 남아있음 → 인접 lang 추론.
        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "digit":
                    if default_lang:
                        temp_item["lang"] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item["lang"] = temp_list[1]["lang"]
                    elif lang_list and i < len(temp_list) - 1:
                        prev_lang = lang_list[-1]["lang"]
                        next_lang = temp_list[i + 1]["lang"]
                        if prev_lang == next_lang:
                            temp_item["lang"] = prev_lang
                        elif lang_list[-1]["text"][-1] in _PUNCT:
                            temp_item["lang"] = next_lang
                        elif temp_list[i + 1]["text"][0] in _PUNCT:
                            temp_item["lang"] = prev_lang
                        elif temp_item["text"][-1] in ("。", "."):
                            temp_item["lang"] = prev_lang
                        elif len(lang_list[-1]["text"]) >= len(temp_list[i + 1]["text"]):
                            temp_item["lang"] = prev_lang
                        else:
                            temp_item["lang"] = next_lang
                    else:
                        temp_item["lang"] = "en"

                lang_list = _merge_lang(lang_list, temp_item)

        return lang_list
