# by https://github.com/Cosmo-klara

from __future__ import print_function

import re
import inflect
import unicodedata

# 접미 계량 단위 치환 테이블
measurement_map = {
    "m": ["meter", "meters"],
    "km": ["kilometer", "kilometers"],
    "km/h": ["kilometer per hour", "kilometers per hour"],
    "ft": ["feet", "feet"],
    "L": ["liter", "liters"],
    "tbsp": ["tablespoon", "tablespoons"],
    "tsp": ["teaspoon", "teaspoons"],
    "h": ["hour", "hours"],
    "min": ["minute", "minutes"],
    "s": ["second", "seconds"],
    "°C": ["degree celsius", "degrees celsius"],
    "°F": ["degree fahrenheit", "degrees fahrenheit"],
}


# 12,000 형식 인식
_inflect = inflect.engine()

# 숫자 서수 변환
_ordinal_number_re = re.compile(r"\b([0-9]+)\. ")

# 숫자 정규식은 \d를 쓰는 게 나을 수도

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")

# 시간 인식
_time_re = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")

# 접미 계량 단위 인식
_measurement_re = re.compile(r"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b")

# 전후 £ 인식 (양쪽 인식을 작성했지만 왜인지 실패함)
_pounds_re_start = re.compile(r"£([0-9\.\,]*[0-9]+)")
_pounds_re_end = re.compile(r"([0-9\.\,]*[0-9]+)£")

# 전후 $ 인식
_dollars_re_start = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_dollars_re_end = re.compile(r"([(0-9\.\,]*[0-9]+)\$")

# 소수 인식
_decimal_number_re = re.compile(r"([0-9]+\.\s*[0-9]+)")

# 분수 인식 (형식 "3/4")
_fraction_re = re.compile(r"([0-9]+/[0-9]+)")

# 서수 인식
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")

# 숫자 처리
_number_re = re.compile(r"[0-9]+")


def _convert_ordinal(m):
    """
    서수 표준화, 예: 1. 2. 3. 4. 5. 6.
    Examples:
        input: "1. "
        output: "1st"
    이후 _expand_ordinal에서 first 등으로 변환
    """
    ordinal = _inflect.ordinal(m.group(1))
    return ordinal + ", "


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_time(m):
    """
    24시간제 시간을 12시간제 표현으로 변환.

    Examples:
        input: "13:00 / 4:00 / 13:30"
        output: "one o'clock p.m. / four o'clock am. / one thirty p.m."
    """
    hours, minutes = map(int, m.group(1, 2))
    period = "a.m." if hours < 12 else "p.m."
    if hours > 12:
        hours -= 12

    hour_word = _inflect.number_to_words(hours)
    minute_word = _inflect.number_to_words(minutes) if minutes != 0 else ""

    if minutes == 0:
        return f"{hour_word} o'clock {period}"
    else:
        return f"{hour_word} {minute_word} {period}"


def _expand_measurement(m):
    """
    일반적인 계량 단위 접미사 처리. 현재 지원: m, km, km/h, ft, L, tbsp, tsp, h, min, s, °C, °F
    확장 시 _measurement_re와 measurement_map 수정
    """
    sign = m.group(3)
    ptr = 1
    # 숫자 추출 방법이 마땅치 않고, 1.2는 어차피 복수 읽기이므로 "."를 제거
    num = int(m.group(1).replace(sign, "").replace(".", ""))
    decimal_part = m.group(2)
    # 위 판단의 예외 (예: 0.1) 처리
    if decimal_part == None and num == 1:
        ptr = 0
    return m.group(1).replace(sign, " " + measurement_map[sign][ptr])


def _expand_pounds(m):
    """
    특별한 규격 명세 없음, 달러 처리와 동일하며 사실 둘을 합칠 수 있음
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"  # Unexpected format
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s and %s %s" % (pounds, pound_unit, pence, penny_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, penny_unit)
    else:
        return "zero pounds"


def _expand_dollars(m):
    """
    change: 센트는 100 한도이므로 0 채우기 필요
    Example:
        input: "32.3$ / $6.24"
        output: "thirty-two dollars and thirty cents" / "six dollars and twenty-four cents"
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s and %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


# 소수 처리
def _expand_decimal_number(m):
    """
    Example:
        input: "13.234"
        output: "thirteen point two three four"
    """
    match = m.group(1)
    parts = match.split(".")
    words = []
    # 문자열의 각 문자 순회
    for char in parts[1]:
        if char == ".":
            words.append("point")
        else:
            words.append(char)
    return parts[0] + " point " + " ".join(words)


# 분수 처리
def _expend_fraction(m):
    """
    규칙1: 분자는 기수 읽기, 분모는 서수 읽기.
    규칙2: 분자가 1보다 크면 분모를 서수 복수로 읽기.
    규칙3: 분모가 2이면 half로 읽고, 분자가 1보다 크면 halves로 읽기.
    Examples:

    | Written |	Said |
    |:---:|:---:|
    | 1/3 | one third |
    | 3/4 | three fourths |
    | 5/6 | five sixths |
    | 1/2 | one half |
    | 3/2 | three halves |
    """
    match = m.group(0)
    numerator, denominator = map(int, match.split("/"))

    numerator_part = _inflect.number_to_words(numerator)
    if denominator == 2:
        if numerator == 1:
            denominator_part = "half"
        else:
            denominator_part = "halves"
    elif denominator == 1:
        return f"{numerator_part}"
    else:
        denominator_part = _inflect.ordinal(_inflect.number_to_words(denominator))
        if numerator > 1:
            denominator_part += "s"

    return f"{numerator_part} {denominator_part}"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


# 사칙연산
RE_ASMD = re.compile(
    r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))\s+([\+\-\×÷=])\s+((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))"
)
# RE_ASMD = re.compile(
#     r"\b((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))([\+\-\×÷=])((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))\b"
# )

asmd_map = {"+": " plus ", "-": " minus ", "×": " times ", "÷": " divided by ", "=": " Equals "}


def replace_asmd(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(1) + asmd_map[match.group(8)] + match.group(9)
    return result


RE_INTEGER = re.compile(r"(?:^|\s+)(-)" r"(\d+)")


def replace_negative_num(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    sign: str = "negative " if sign else ""
    result = f"{sign}{number}"
    return result



def normalize(text):
    """
    !!! 모든 처리는 올바른 입력이 필요 !!!
    새로운 처리 추가 시 정규식과 대응하는 처리 함수만 추가하면 됨
    """

    text = re.sub(_ordinal_number_re, _convert_ordinal, text)

    # 수학 연산 처리
    # 대체: text = re.sub(r"(?<!\d)-|-(?!\d)", " minus ", text)
    while RE_ASMD.search(text):
        text = RE_ASMD.sub(replace_asmd, text)
    text = RE_INTEGER.sub(replace_negative_num, text)

    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_pounds_re_start, _expand_pounds, text)
    text = re.sub(_pounds_re_end, _expand_pounds, text)
    text = re.sub(_dollars_re_start, _expand_dollars, text)
    text = re.sub(_dollars_re_end, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_fraction_re, _expend_fraction, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)

    text = "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )  # Strip accents

    text = re.sub("%", " percent", text)
    text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)
    # 순수 대문자 단어 분리 추가
    text = re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
    return text
