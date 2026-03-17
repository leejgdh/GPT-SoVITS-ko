# -*- coding: utf-8 -*-
"""오디오 슬라이싱 CLI.

긴 오디오 파일을 무음 구간 기준으로 짧은 세그먼트로 분할한다.
32kHz 샘플링, 볼륨 정규화 적용.

출력:
  - {output-dir}/{filename}_{start}_{end}.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# ---------------------------------------------------------------

import numpy as np
from loguru import logger
from scipy.io import wavfile

from tools.my_utils import load_audio
from tools.slicer2 import Slicer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="무음 기반 오디오 슬라이싱")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--inp", default=None, help="입력 오디오 파일 또는 폴더 (기본: {voice-dir}/step1/01_denoise)")
    parser.add_argument("--output-dir", default=None, help="출력 폴더 (기본: {voice-dir}/step1/02_sliced)")
    parser.add_argument("--threshold", type=int, default=-34, help="무음 판정 볼륨 임계값 dB (기본: -34)")
    parser.add_argument("--min-length", type=int, default=4000, help="최소 세그먼트 길이 ms (기본: 4000)")
    parser.add_argument("--min-interval", type=int, default=300, help="최소 무음 간격 ms (기본: 300)")
    parser.add_argument("--hop-size", type=int, default=10, help="볼륨 곡선 hop size ms (기본: 10)")
    parser.add_argument("--max-sil-kept", type=int, default=500, help="세그먼트 앞뒤 최대 무음 유지 ms (기본: 500)")
    parser.add_argument("--max", type=float, default=0.9, help="정규화 최대값 (기본: 0.9)")
    parser.add_argument("--alpha", type=float, default=0.25, help="정규화 믹싱 비율 (기본: 0.25)")
    parser.add_argument("--i-part", type=int, default=0, help="파티션 인덱스 (분산 처리용)")
    parser.add_argument("--all-parts", type=int, default=1, help="총 파티션 수")
    args = parser.parse_args()
    if args.inp is None:
        args.inp = os.path.join(args.voice_dir, "step1", "01_denoise")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.voice_dir, "step1", "02_sliced")
    return args


def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.inp):
        input_files = [args.inp]
    elif os.path.isdir(args.inp):
        input_files = [os.path.join(args.inp, name) for name in sorted(os.listdir(args.inp))]
    else:
        logger.error("입력 경로가 파일도 폴더도 아닙니다: {}", args.inp)
        return

    slicer = Slicer(
        sr=32000,
        threshold=args.threshold,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )

    _max = args.max
    alpha = args.alpha

    for inp_path in input_files[args.i_part :: args.all_parts]:
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    os.path.join(args.output_dir, f"{name}_{start:010d}_{end:010d}.wav"),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", os.path.basename(inp_path), e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())

    logger.info("슬라이싱 완료.")


if __name__ == "__main__":
    main()
