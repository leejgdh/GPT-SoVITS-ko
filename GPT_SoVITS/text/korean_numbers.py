"""한국어 문장 안의 아라비아 숫자를 읽는 소리대로 한글로 풀어 쓴다.

숫자 뒤 단위명사와 앞뒤 형태소로 고유어 / 한자어 읽기를 정한다.
날짜(2026.10.18) · 시각(10:30) · 전화번호 · 소수 · 범위(2~3개, 2에서 3개)를 함께 다룬다.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_SINO_DIGITS = "영일이삼사오육칠팔구"
_DIGIT_CALLS = "공일이삼사오육칠팔구"
_NATIVE_UNITS = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
_SMALL_UNITS = ("", "십", "백", "천")
_LARGE_UNITS = ("", "만", "억", "조", "경")

# 고유어 수로 읽는 단위명사. 99 까지 고유어, 100 이상은 백 자리 위만 한자어로 읽는다.
_NATIVE_COUNTERS = frozenset(
    "개 명 사람 마리 살 시간 가지 곳 군데 잔 병 그릇 벌 켤레 송이 달 줄 차례 채 척 통 그루 자루 "
    "포기 알 판 개비 모금 마디 바퀴 걸음 방울 봉지 상자 접시 끼 밤 표 쌍 팀 곡 문장 단어 글자 "
    "발 컵 손 덩이 조각 묶음 다발 뿌리 숟가락 숟갈 스푼 첩 톨 닢 움큼".split()
)
# 앞에 '제' 가 붙으면 차례(한자어), 아니면 수량(고유어)으로 읽는 단위명사.
_ORDINAL_PREFIXED_COUNTERS = frozenset("장 권 편".split())
# '번' 의 앞이나 뒤에 오면 횟수가 아니라 번호로 읽게 하는 명사.
_NUMBERED_NOUNS = frozenset(
    "버스 출구 문제 질문 문항 보기 선택지 선 호선 노선 국도 트랙 채널 게이트 타자 타석 홈 승강장 "
    "좌석 자리 항목 번호 사진 카드 레인 코스 플랫폼".split()
)
# '대' 가 10 단위 수(10대, 20대)와 붙어도 연령대가 아니라 대수로 읽게 하는 앞 명사.
_MACHINE_NOUNS = frozenset(
    "차 차량 자동차 버스 트럭 택시 오토바이 자전거 비행기 헬기 기계 컴퓨터 노트북 서버 카메라 "
    "드론 로봇 장비".split()
)
# '분' 앞에 오면 사람을 세는 '분' 으로 읽게 하는 명사.
_HONORED_NOUNS = frozenset("손님 어르신 고객 선생님 부모님".split())
# '분' 바로 뒤에서 사람을 세는 '분' 의 주어 · 목적어 자리를 나타내는 조사.
_HONORED_PARTICLES = frozenset("이 께서 이서 께 을 과 와 은".split())
_HONORIFIC_ENDINGS = ("시", "셨", "세", "셔")
_HONORIFIC_VERBS = ("계시", "계셨", "계세", "계셔", "모시", "모셨", "모셔")
_HONORIFIC_SCAN = 4
# 'N째' 의 1~4 는 관형형이 아닌 수사 형태로 읽는다 (첫째 · 둘째 · 셋째 · 넷째).
_ORDINAL_SUFFIX_FORMS = {1: "첫", 2: "둘", 3: "셋", 4: "넷"}

_DATE_RE = re.compile(r"(?<![\d.])(\d{4})([./-])(\d{1,2})\2(\d{1,2})(?![\d.])")
_TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_PHONE_RE = re.compile(r"(?<![\d-])0\d{1,2}-\d{3,4}-\d{4}(?![\d-])")
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_RANGE_SEP_RE = re.compile(r"[ \t]*(에서|~)[ \t]*")


class _Morpheme(Protocol):
    surface: str
    pos: str
    span: "_Span"


class _Span(Protocol):
    start: int
    end: int


class MorphAnalyzer(Protocol):
    """형태소 분석기 — ``parse`` 가 문자 offset(span) 이 붙은 형태소 목록을 돌려준다."""

    def parse(self, text: str) -> Sequence[_Morpheme]: ...


@dataclass
class _Replacement:
    start: int
    end: int
    text: str


@dataclass
class _NumberToken:
    start: int
    end: int
    raw: str
    counter: _Morpheme | None
    prev: _Morpheme | None
    after: list[_Morpheme]


def spell_sino(value: int) -> str:
    """정수를 한자어 수로 쓴다 (12345 → 만이천삼백사십오). 만 단위로 띄어 쓴다."""
    if value == 0:
        return "영"
    groups: list[str] = []
    for index, unit in enumerate(_LARGE_UNITS):
        group = value // (10_000**index) % 10_000
        if group == 0:
            continue
        if group == 1 and unit == "만" and value < 100_000:
            groups.append("만")
            continue
        groups.append(_spell_sino_group(group) + unit)
    return " ".join(reversed(groups))


def _spell_sino_group(value: int) -> str:
    """0~9999 를 한자어 수로 쓴다. 십 · 백 · 천 앞의 '일' 은 뺀다."""
    words: list[str] = []
    for place in range(3, -1, -1):
        digit = value // (10**place) % 10
        if digit == 0:
            continue
        if digit == 1 and place > 0:
            words.append(_SMALL_UNITS[place])
        else:
            words.append(_SINO_DIGITS[digit] + _SMALL_UNITS[place])
    return "".join(words)


def spell_native(value: int) -> str:
    """정수를 단위명사 앞 고유어 관형형으로 쓴다 (2 → 두, 20 → 스무, 123 → 백스물세).

    100 이상은 백 자리 위를 한자어로, 아래 두 자리를 고유어로 쓴다. 0 은 '영' 이다.
    """
    if value == 0:
        return "영"
    upper, lower = divmod(value, 100)
    head = spell_sino(upper * 100) if upper else ""
    if lower == 0:
        return head
    if lower == 20:
        return head + "스무"
    return head + _NATIVE_TENS[lower // 10] + _NATIVE_UNITS[lower % 10]


def _spell_digits(digits: str, *, zero: str) -> str:
    """숫자열을 한 자리씩 읽는다. 0 은 ``zero`` 로 읽는다."""
    names = zero + _SINO_DIGITS[1:]
    return "".join(names[int(d)] for d in digits)


def _spell_decimal(raw: str) -> str:
    """소수를 읽는다 (3.05 → 삼 점 영오)."""
    integer, fraction = raw.replace(",", "").split(".")
    return f"{spell_sino(int(integer))} 점 {_spell_digits(fraction, zero='영')}"


def _spell_month(value: int) -> str:
    """'월' 앞의 수를 쓴다 (6 → 유, 10 → 시)."""
    return {6: "유", 10: "시"}.get(value, spell_sino(value))


def _spell_hour(value: int) -> str:
    """'시' 앞의 수를 쓴다 — 1~12 는 고유어, 그 밖은 한자어."""
    return spell_native(value) if 1 <= value <= 12 else spell_sino(value)


def _is_numbering(prev: _Morpheme | None, after: list[_Morpheme]) -> bool:
    """'N번' 이 횟수가 아니라 번호인지 — 앞이 '제' 이거나 앞뒤에 번호를 붙이는 명사가 있다."""
    if prev is not None and (prev.pos == "XPN" or prev.surface in _NUMBERED_NOUNS):
        return True
    return bool(after) and after[0].surface in _NUMBERED_NOUNS


def _is_honored_person(prev: _Morpheme | None, after: list[_Morpheme]) -> bool:
    """'N분' 이 시간이 아니라 사람을 세는 말인지 — 존칭 명사 · 조사 + 높임 서술어로 판정한다."""
    if prev is not None and prev.surface in _HONORED_NOUNS:
        return True
    if not after or after[0].surface not in _HONORED_PARTICLES:
        return False
    if after[0].surface in ("께서", "이서", "께"):
        return True
    for morph in after[1 : 1 + _HONORIFIC_SCAN]:
        if "EP" in morph.pos and morph.surface.startswith(_HONORIFIC_ENDINGS):
            return True
        if morph.pos.startswith("VV") and morph.surface.startswith(_HONORIFIC_VERBS):
            return True
    return False


def _spell_counted(token: _NumberToken) -> str:
    """단위명사가 붙은 정수를 그 단위의 읽기 규칙대로 쓴다."""
    value = int(token.raw.replace(",", ""))
    counter = token.counter.surface if token.counter is not None else ""
    prev, after = token.prev, token.after
    if counter == "째":
        return _ORDINAL_SUFFIX_FORMS.get(value) or spell_native(value)
    if counter == "번" and after and after[0].surface == "째":
        return "첫" if value == 1 else spell_native(value)
    if counter == "번":
        return spell_sino(value) if _is_numbering(prev, after) else spell_native(value)
    if counter == "월":
        return _spell_month(value)
    if counter == "시":
        return _spell_hour(value)
    if counter == "분":
        return spell_native(value) if _is_honored_person(prev, after) else spell_sino(value)
    if counter in _ORDINAL_PREFIXED_COUNTERS:
        return spell_sino(value) if prev is not None and prev.pos == "XPN" else spell_native(value)
    if counter == "대":
        if after and after[0].pos == "SN":
            return spell_sino(value)
        is_decade = value % 10 == 0 and 10 <= value <= 90
        if is_decade and (prev is None or prev.surface not in _MACHINE_NOUNS):
            return spell_sino(value)
        return spell_native(value)
    if counter == "배":
        return spell_native(value) if value < 10 else spell_sino(value)
    if counter in _NATIVE_COUNTERS:
        return spell_native(value)
    return spell_sino(value)


def _spell_number(token: _NumberToken) -> str:
    """숫자 토큰 하나를 읽는 소리대로 쓴다."""
    plain = token.raw.replace(",", "")
    if "." in plain:
        return _spell_decimal(token.raw)
    if len(plain) > 1 and plain.startswith("0"):
        return _spell_digits(plain, zero="공")
    if len(plain) > 20:
        return _spell_digits(plain, zero="영")
    return _spell_counted(token)


def _date_replacements(text: str) -> list[_Replacement]:
    """'2026.10.18' 꼴 날짜를 '이천이십육 년 시월 십팔 일' 로 바꿀 자리를 찾는다."""
    found: list[_Replacement] = []
    for match in _DATE_RE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(3)), int(match.group(4))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        spoken = f"{spell_sino(year)}년 {_spell_month(month)}월 {spell_sino(day)}일"
        found.append(_Replacement(match.start(), match.end(), spoken))
    return found


def _time_replacements(text: str) -> list[_Replacement]:
    """'10:30' 꼴 시각을 '열 시 삼십 분' 으로 바꿀 자리를 찾는다."""
    found: list[_Replacement] = []
    for match in _TIME_RE.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 24 or minute > 59:
            continue
        spoken = f"{_spell_hour(hour)}시"
        if minute:
            spoken += f" {spell_sino(minute)}분"
        found.append(_Replacement(match.start(), match.end(), spoken))
    return found


def _phone_replacements(text: str) -> list[_Replacement]:
    """전화번호를 한 자리씩 읽고 0 은 '공' 으로 읽을 자리를 찾는다."""
    return [
        _Replacement(
            match.start(),
            match.end(),
            ", ".join(_spell_digits(group, zero="공") for group in match.group().split("-")),
        )
        for match in _PHONE_RE.finditer(text)
    ]


def _overlaps(start: int, end: int, taken: list[_Replacement]) -> bool:
    return any(start < r.end and r.start < end for r in taken)


def _number_tokens(text: str, morphs: Sequence[_Morpheme], taken: list[_Replacement]) -> list[_NumberToken]:
    """이미 바꿀 자리로 잡힌 구간을 뺀 숫자마다 앞 형태소 · 단위 형태소 · 뒤 형태소를 묶는다."""
    by_start = {m.span.start: i for i, m in enumerate(morphs)}
    by_end = {m.span.end: i for i, m in enumerate(morphs)}
    tokens: list[_NumberToken] = []
    for match in _NUMBER_RE.finditer(text):
        if _overlaps(match.start(), match.end(), taken):
            continue
        start, end = match.start(), match.end()
        prev_end = start - 1 if start > 0 and text[start - 1] == " " else start
        prev_index = by_end.get(prev_end)
        next_start = end + 1 if end < len(text) and text[end] == " " else end
        counter_index = by_start.get(next_start)
        counter: _Morpheme | None = None
        after: list[_Morpheme] = []
        if counter_index is not None and morphs[counter_index].pos.startswith(("NN", "NR", "XSN")):
            counter = morphs[counter_index]
            after = list(morphs[counter_index + 1 :])
        prev = morphs[prev_index] if prev_index is not None and morphs[prev_index].pos != "SN" else None
        tokens.append(_NumberToken(start, end, match.group(), counter, prev, after))
    return tokens


def _range_replacements(text: str, tokens: list[_NumberToken]) -> list[_Replacement]:
    """'2~3개' · '2에서 3개' 의 앞 숫자를 뒤 숫자의 단위로 읽고 단위를 붙인다 (두개에서 세개)."""
    found: list[_Replacement] = []
    for first, second in zip(tokens, tokens[1:]):
        if first.counter is not None or second.counter is None:
            continue
        sep = _RANGE_SEP_RE.fullmatch(text, first.end, second.start)
        if sep is None or "." in first.raw:
            continue
        borrowed = _NumberToken(first.start, first.end, first.raw, second.counter, first.prev, second.after)
        spoken = f"{_spell_number(borrowed)}{second.counter.surface}에서 "
        found.append(_Replacement(first.start, second.start, spoken))
    return found


def normalize_numbers(text: str, analyzer: MorphAnalyzer) -> str:
    """문장 안의 아라비아 숫자를 읽는 소리대로 한글로 바꾼다.

    단위명사 · 범위 단위는 원문 그대로 두고 숫자 자리만 바꾼다. 숫자가 없으면 원문을 돌려준다.
    """
    if not any(ch.isdigit() for ch in text):
        return text
    replacements = _date_replacements(text)
    replacements += [r for r in _time_replacements(text) if not _overlaps(r.start, r.end, replacements)]
    replacements += [r for r in _phone_replacements(text) if not _overlaps(r.start, r.end, replacements)]
    tokens = _number_tokens(text, analyzer.parse(text), replacements)
    ranges = _range_replacements(text, tokens)
    replacements += ranges
    for token in tokens:
        if not _overlaps(token.start, token.end, ranges):
            replacements.append(_Replacement(token.start, token.end, _spell_number(token)))
    for item in sorted(replacements, key=lambda r: r.start, reverse=True):
        text = text[: item.start] + item.text + text[item.end :]
    return text
