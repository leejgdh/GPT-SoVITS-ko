import re


def str2bool(str):
    return True if str.lower() == "true" else False


def get_newest_ckpt(string_list):
    # 문자열의 숫자를 매칭하기 위한 정규식 패턴
    pattern = r"epoch=(\d+)-step=(\d+)\.ckpt"

    # 정규식으로 각 문자열의 숫자 정보를 추출하여 튜플 리스트 생성
    extracted_info = []
    for string in string_list:
        match = re.match(pattern, string)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            extracted_info.append((epoch, step, string))
    # epoch과 step 숫자 기준으로 정렬
    sorted_info = sorted(extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
    # 최신 체크포인트 파일명 반환
    newest_ckpt = sorted_info[0][2]
    return newest_ckpt


# 텍스트가 존재하고 비어있지 않으면 True 반환
def check_txt_file(file_path):
    try:
        with open(file_path, "r") as file:
            text = file.readline().strip()
        assert text.strip() != ""
        return text
    except Exception:
        return False
    return False
