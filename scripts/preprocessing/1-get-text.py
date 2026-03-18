# -*- coding: utf-8 -*-
"""텍스트 → 음소 추출 (전처리 1단계).

입력 텍스트 파일(wav_name|speaker|language|text)을 읽어
음소 시퀀스를 추출한다.

출력:
  - {opt_dir}/name2text-{i_part}.txt  (음소 + word2ph + 정규화 텍스트)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from loguru import logger

# -- 경로 부트스트랩 (프로젝트 내부 import 전에 실행) --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import filter_label_lines, parse_label_line, setup_paths

setup_paths()
# ---------------------------------------------------------------

from text.cleaner import clean_text

from tools.utils.audio import clean_path

_LANG_MAP: dict[str, str] = {
    "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
    "EN": "en", "en": "en", "En": "en",
    "KO": "ko", "Ko": "ko", "ko": "ko",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="텍스트 → 음소 추출")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--inp-text", default=None, help="입력 텍스트 파일 (기본: {voice-dir}/step1/04_asr/03_vocal.list)")
    parser.add_argument("--opt-dir", default=None, help="출력 디렉토리 (기본: {voice-dir}/step2)")
    parser.add_argument("--i-part", type=int, default=0, help="파티션 인덱스 (분산 처리용)")
    parser.add_argument("--all-parts", type=int, default=1, help="총 파티션 수")
    parser.add_argument("--version", default="v2Pro", help="모델 버전 (기본: v2Pro)")
    args = parser.parse_args()
    if args.inp_text is None:
        args.inp_text = os.path.join(args.voice_dir, "step1", "04_asr", "03_vocal.list")
    if args.opt_dir is None:
        args.opt_dir = os.path.join(args.voice_dir, "step2", args.version)
    return args


def main() -> None:
    args = _parse_args()

    txt_path = os.path.join(args.opt_dir, f"name2text-{args.i_part}.txt")
    os.makedirs(args.opt_dir, exist_ok=True)

    def _extract_phonemes(items: list, results: list) -> None:
        for name, text, lang in items:
            try:
                name = clean_path(name)
                name = os.path.basename(name)
                logger.debug("{}", name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lang, args.version,
                )
                phones = " ".join(phones)
                results.append([name, phones, word2ph, norm_text])
            except Exception as e:
                logger.warning("{} -> 건너뜀 ({})", name, e)
                logger.debug("상세 traceback:\n{}", traceback.format_exc())

    # 입력 파일 읽기 + 상태 필터링 + 파티셔닝
    with open(args.inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines = filter_label_lines(lines)

    pending: list = []
    for line in lines[args.i_part :: args.all_parts]:
        try:
            wav_name, spk_name, language, text = parse_label_line(line)
            text = text.strip()
            if not text:
                continue
            if language in _LANG_MAP:
                pending.append([wav_name, text, _LANG_MAP[language]])
            else:
                logger.warning("The language={} of {} is not supported for training.", language, wav_name)
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", line.split("|")[0], e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())

    results: list = []
    _extract_phonemes(pending, results)

    output_lines = []
    for name, phones, word2ph, norm_text in results:
        output_lines.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
