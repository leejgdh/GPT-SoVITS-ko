# -*- coding: utf-8 -*-
"""Faster Whisper 음성 인식 라벨링 CLI.

Faster Whisper 모델을 사용하여 오디오 파일에서 텍스트를 추출하고
학습용 라벨 파일을 생성한다. 한국어, 영어, 일본어 지원.

출력:
  - {output-folder}/{폴더명}.list  (filepath|speaker|lang|text|state 형식)
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
import torch
from faster_whisper import WhisperModel
from loguru import logger
from tqdm import tqdm

from tools.asr.config import get_models
from tools.utils.download import download_whisper_model
from tools.utils.audio import load_audio, load_cudnn

_LANGUAGE_CODES = ["ko", "en", "ja", "auto"]
_AUDIO_SR = 32000


def _check_audio_quality(file_path: str, min_duration: float, min_rms: float) -> str | None:
    """오디오 파일의 품질을 검증한다.

    Returns:
        None이면 정상, 문자열이면 rejected 사유.
    """
    try:
        audio = load_audio(file_path, _AUDIO_SR)
    except Exception:
        return "로드 실패"
    if not np.isfinite(audio).all():
        return "NaN/Inf 포함"
    duration = len(audio) / _AUDIO_SR
    if duration < min_duration:
        return f"너무 짧음 ({duration:.2f}s)"
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < min_rms:
        return f"무음 (RMS={rms:.4f})"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faster Whisper 음성 인식 라벨링")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("-i", "--input-folder", default=None, help="입력 오디오 폴더 (기본: {voice-dir}/step1/03_vocal)")
    parser.add_argument("-o", "--output-folder", default=None, help="라벨 파일 출력 폴더 (기본: {voice-dir}/step1/04_asr)")
    parser.add_argument(
        "-s", "--model-size", default="large-v3", choices=get_models(),
        help="Whisper 모델 크기 (기본: large-v3)",
    )
    parser.add_argument(
        "-l", "--language", default="ko", choices=_LANGUAGE_CODES,
        help="오디오 언어 (기본: ko, auto=자동 감지)",
    )
    parser.add_argument(
        "-p", "--precision", default="float16", choices=["float16", "float32", "int8"],
        help="연산 정밀도 (기본: float16)",
    )
    parser.add_argument("--min-duration", type=float, default=0.3, help="최소 오디오 길이 초 — 미달 시 rejected (기본: 0.3)")
    parser.add_argument("--min-rms", type=float, default=0.01, help="최소 RMS — 미달 시 무음으로 rejected (기본: 0.01)")
    args = parser.parse_args()
    if args.input_folder is None:
        args.input_folder = os.path.join(args.voice_dir, "step1", "03_vocal")
    if args.output_folder is None:
        args.output_folder = os.path.join(args.voice_dir, "step1", "04_asr")
    return args


def main() -> None:
    args = _parse_args()

    load_cudnn()

    model_size = args.model_size
    if model_size == "large":
        model_size = "large-v3"
    model_path = download_whisper_model(model_size)

    language = args.language if args.language != "auto" else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Faster Whisper 모델 로딩: {}", model_path)
    model = WhisperModel(model_path, device=device, compute_type=args.precision)

    input_file_names = sorted(os.listdir(args.input_folder))
    output = []
    output_file_name = os.path.basename(args.input_folder)
    rejected_count = 0

    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(args.input_folder, file_name)

            # rule-based 품질 검증 (빠른 필터링)
            quality_issue = _check_audio_quality(file_path, args.min_duration, args.min_rms)
            if quality_issue:
                logger.warning("{} -> rejected ({})", file_name, quality_issue)
                output.append(f"{file_path}|{output_file_name}|||rejected")
                rejected_count += 1
                continue

            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
            )
            text = ""
            for segment in segments:
                text += segment.text

            if text.strip():
                state = "pending"
            else:
                state = "rejected"
                rejected_count += 1

            output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}|{state}")
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", file_name, e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())

    os.makedirs(args.output_folder, exist_ok=True)
    output_file_path = os.path.abspath(os.path.join(args.output_folder, f"{output_file_name}.list"))

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    if rejected_count:
        logger.info("자동 rejected {} 건 (품질 미달 또는 빈 텍스트)", rejected_count)
    logger.info("ASR 완료 -> 라벨 파일: {} (총 {} 건)", output_file_path, len(output))


if __name__ == "__main__":
    main()
