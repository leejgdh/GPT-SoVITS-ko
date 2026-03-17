import os
import sys
import threading

from loguru import logger
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

punctuation = set(["!", "?", "…", ",", ".", "-"])


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class TextPreprocessor:
    def __init__(self, device: torch.device):
        self.device = device

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        logger.info("텍스트 분할")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        logger.info("텍스트 음소 변환")
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        logger.debug("실제 입력된 목표 텍스트: {}", text)

        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        logger.debug("실제 입력된 목표 텍스트(분절 후): {}", texts)
        return texts

    def segment_and_extract_feature_for_text(
        self, text: str, language: str, version: str = "v1"
    ) -> Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        text = re.sub(r' {2,}', ' ', text)
        textlist = []
        langlist = []
        if language == "all_ja":
            for tmp in LangSegmenter.getTexts(text, "ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text, "ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])

        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert, norm_text

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        feature = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float32,
        ).to(self.device)
        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("유효한 텍스트를 입력하세요")
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result
