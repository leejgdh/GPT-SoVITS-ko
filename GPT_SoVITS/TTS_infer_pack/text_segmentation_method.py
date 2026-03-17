import re
from typing import Callable

punctuation = set(["!", "?", "…", ",", ".", "-", " "])
METHODS = dict()


def get_method(name: str) -> Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method


def get_method_names() -> list:
    return list(METHODS.keys())


def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func

    return decorator


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def split_big_text(text, max_len=510):
    # 전각 및 반각 구두점 정의
    punctuation = "".join(splits)

    # 텍스트 분할
    segments = re.split("([" + punctuation + "])", text)

    # 결과 리스트 및 현재 세그먼트 초기화
    result = []
    current_segment = ""

    for segment in segments:
        # 현재 세그먼트에 새 세그먼트를 더하면 max_len 초과 시, 현재 세그먼트를 결과에 추가하고 초기화
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 마지막 세그먼트를 결과 리스트에 추가
    if current_segment:
        result.append(current_segment)

    return result


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 끝에는 반드시 구두점이 있으므로 바로 종료, 마지막 단락은 이전에 추가됨
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


# 자르지 않음
@register_method("cut0")
def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "/n"


# 4문장씩 묶어 자르기
@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 50자씩 묶어 자르기
@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ## 마지막 항목이 너무 짧으면 이전 항목과 합침
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 마침표(。)로 자르기
@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 마침표(.)로 자르기
@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# 구두점으로 자르기
# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
@register_method("cut5")
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)
