# -*- coding: utf-8 -*-
"""오디오 노이즈 제거 CLI.

ModelScope의 FRCRN 모델을 사용하여 오디오에서 배경 잡음을 제거한다.
모델: speech_frcrn_ans_cirm_16k (DAMO Academy)

출력:
  - {output-folder}/ (노이즈 제거된 오디오)
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

import tempfile

import numpy as np
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from scipy.io import wavfile
from tqdm import tqdm

from tools.utils.audio import load_audio

_DEFAULT_MODEL_PATH = "data/models/denoise/speech_frcrn_ans_cirm_16k"
_FALLBACK_MODEL_ID = "damo/speech_frcrn_ans_cirm_16k"
_TARGET_SR = 16000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FRCRN 오디오 노이즈 제거")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("-i", "--input-folder", default=None, help="입력 오디오 폴더 (기본: {voice-dir}/raw_audio)")
    parser.add_argument("-o", "--output-folder", default=None, help="출력 폴더 (기본: {voice-dir}/step1/01_denoise)")
    parser.add_argument(
        "--model-path", default=None,
        help=f"FRCRN 모델 경로 (기본: {_DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()
    if args.input_folder is None:
        args.input_folder = os.path.join(args.voice_dir, "raw_audio")
    if args.output_folder is None:
        args.output_folder = os.path.join(args.voice_dir, "step1", "01_denoise")
    return args


def _to_wav16k(inp_path: str, tmp_dir: str) -> str:
    """어떤 포맷이든 16kHz 모노 WAV로 변환한다."""
    audio = load_audio(inp_path, _TARGET_SR)
    stem = os.path.splitext(os.path.basename(inp_path))[0]
    tmp_path = os.path.join(tmp_dir, f"{stem}.wav")
    wavfile.write(tmp_path, _TARGET_SR, (audio * 32767).astype(np.int16))
    return tmp_path


def _resolve_model_path(user_path: str | None) -> str:
    # 우선순위: --model-path → 프로젝트 로컬 → modelscope hub 캐시 → model_id (자동 다운로드).
    # 캐시가 있어도 model_id 로 호출하면 modelscope 가 hub 메타데이터 동기화를 시도해 정체될 수 있어,
    # 로컬 절대 경로를 직접 넘긴다.
    if user_path:
        return user_path
    if os.path.exists(_DEFAULT_MODEL_PATH):
        return _DEFAULT_MODEL_PATH
    cache_root = os.environ.get("MODELSCOPE_CACHE") or os.path.expanduser("~/.cache/modelscope/hub")
    cached = os.path.join(cache_root, "models", _FALLBACK_MODEL_ID)
    if os.path.isdir(cached):
        return cached
    return _FALLBACK_MODEL_ID


def main() -> None:
    args = _parse_args()

    model_path = _resolve_model_path(args.model_path)
    logger.info("FRCRN 모델 경로: {}", model_path)

    ans = pipeline(Tasks.acoustic_noise_suppression, model=model_path)

    os.makedirs(args.output_folder, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in tqdm(os.listdir(args.input_folder)):
            inp_path = os.path.join(args.input_folder, name)
            if not os.path.isfile(inp_path):
                continue
            out_name = os.path.splitext(name)[0] + ".wav"
            out_path = os.path.join(args.output_folder, out_name)
            try:
                wav_path = _to_wav16k(inp_path, tmp_dir)
                ans(wav_path, output_path=out_path)
            except Exception as e:
                logger.warning("{} -> 건너뜀 ({})", name, e)
                logger.debug("상세 traceback:\n{}", traceback.format_exc())


if __name__ == "__main__":
    main()
