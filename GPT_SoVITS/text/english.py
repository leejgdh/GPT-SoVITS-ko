import pickle
import os
import re
import wordsegment
from g2p_en import G2p
from loguru import logger

from text.symbols import punctuation

from text.symbols2 import symbols

from builtins import str as unicode
from text.en_normalization.expend import normalize
from nltk.tokenize import TweetTokenizer

word_tokenize = TweetTokenizer().tokenize
from nltk import pos_tag

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
NAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")


# g2p_en 구두점 호환 처리
rep_map = {
    "[;:：，；]": ",",
    '["’]': "'",
    "。": ".",
    "！": "!",
    "？": "?",
}


arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            logger.warning("ph not in symbols: {}", ph)
    return phs_new


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}\s])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                # 사용자 정의 발음 단어로 사전 덮어쓰기
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict


def text_normalize(text):
    # todo: eng text normalize

    # 동일 효과, 일관성 유지
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    text = pattern.sub(lambda x: rep_map[x.group()], text)

    text = unicode(text)
    text = normalize(text)

    # 중복 구두점으로 인한 참조 누출 방지
    text = replace_consecutive_punctuation(text)
    return text


class en_G2p(G2p):
    def __init__(self):
        super().__init__()
        # 토크나이저 초기화
        wordsegment.load()

        # 구식 사전 확장, 인명 사전 추가
        self.cmu = get_dict()
        self.namedict = get_namedict()

        # 발음 오류가 있는 약어 제거
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # 다음자(多音字) 수정
        self.homograph2features["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (
            ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            "JJ",
        )

    def __call__(self, text):
        # tokenization
        words = word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for o_word, pos in tokens:
            # g2p_en 소문자 변환 로직 복원
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            # 단일 문자 먼저 처리
            elif len(word) == 1:
                # 단독 A 발음 수정, 원본 o_word로 대문자 판별
                if o_word == "A":
                    pron = ["EY1"]
                else:
                    pron = self.cmu[word][0]
            # g2p_en 원본 다음자 처리
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                # pos1이 pos보다 긴 경우는 read에서만 발생
                elif len(pos) < len(pos1) and pos == pos1[: len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                # 재귀적 예측 조회
                pron = self.qryword(o_word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def qryword(self, o_word):
        word = o_word.lower()

        # 사전 조회 (단일 문자 제외)
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # 첫 글자만 대문자인 단어는 인명 사전 조회
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # OOV 길이 3 이하이면 철자 읽기
        if len(word) <= 3:
            phones = []
            for w in word:
                # 단독 A 발음 수정 (대문자 경우 없음)
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # 소유격 분리 시도
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH 무성 자음 뒤 's → ['S']
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            # S Z SH ZH CH JH 마찰음 뒤 's → ['IH1', 'Z'] 또는 ['AH0', 'Z']
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            # B D G DH V M N NG L R W Y 유성 자음 뒤 's → ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IY0 IY1 IY2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 모음 뒤 's → ['Z']
            else:
                phones.extend(["Z"])
            return phones

        # 복합어 처리를 위한 분리 시도
        comps = wordsegment.segment(word.lower())

        # 분리 불가능한 단어는 예측으로 전달
        if len(comps) == 1:
            return self.predict(word)

        # 분리 가능한 단어는 재귀 처리
        return [phone for comp in comps for phone in self.qryword(comp)]


_g2p = en_G2p()


def g2p(text):
    # g2p_en 전체 추론, 존재하지 않는 ARPA 제거 후 반환
    phone_list = _g2p(text)
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]

    return replace_phs(phones)
