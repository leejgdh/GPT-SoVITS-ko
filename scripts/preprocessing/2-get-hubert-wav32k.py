# -*- coding: utf-8 -*-
"""HuBERT 임베딩 + 32kHz 오디오 추출 (전처리 2단계-a).

입력 오디오에서 CNHuBERT 피처를 추출하고,
32kHz로 정규화된 오디오를 저장한다.

출력:
  - {opt_dir}/01_cnhubert/{wav_name}.pt  (HuBERT 임베딩)
  - {opt_dir}/02_wav32k/{wav_name}       (32kHz 정규화 오디오)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from time import time

from loguru import logger

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import filter_label_lines, parse_label_line, setup_paths

setup_paths()
# ---------------------------------------------------------------

import librosa
import numpy as np
import torch
from feature_extractor import cnhubert
from scipy.io import wavfile

from tools.dl_utils import ensure_dir
from tools.my_utils import clean_path, load_audio

_DEFAULT_CNHUBERT_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
_CNHUBERT_HF_REPO = "TencentGameMate/chinese-hubert-base"


def _save_tensor(tensor: torch.Tensor, path: str, i_part: int) -> None:
    """torch.save가 비ASCII 경로를 지원하지 않는 문제 우회."""
    dir_ = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth" % (time(), i_part)
    torch.save(tensor, tmp_path)
    shutil.move(tmp_path, os.path.join(dir_, name))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HuBERT 임베딩 + 32kHz 오디오 추출")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--inp-text", default=None, help="입력 텍스트 파일 (기본: {voice-dir}/step1/04_asr/03_vocal.list)")
    parser.add_argument("--opt-dir", default=None, help="출력 디렉토리 (기본: {voice-dir}/step2)")
    parser.add_argument(
        "--cnhubert-dir", default=_DEFAULT_CNHUBERT_DIR,
        help=f"CNHuBERT pretrained 모델 경로 (기본: {_DEFAULT_CNHUBERT_DIR})",
    )
    parser.add_argument("--i-part", type=int, default=0, help="파티션 인덱스 (분산 처리용)")
    parser.add_argument("--all-parts", type=int, default=1, help="총 파티션 수")
    parser.add_argument("--no-half", action="store_true", help="FP32 사용 (기본: FP16)")
    parser.add_argument("--version", default="v2Pro", help="모델 버전 (기본: v2Pro)")
    args = parser.parse_args()
    if args.inp_text is None:
        args.inp_text = os.path.join(args.voice_dir, "step1", "04_asr", "03_vocal.list")
    if args.opt_dir is None:
        args.opt_dir = os.path.join(args.voice_dir, "step2", args.version)
    return args


def main() -> None:
    args = _parse_args()

    is_half = (not args.no_half) and torch.cuda.is_available()

    # CNHuBERT 모델 자동 다운로드 + 경로 설정
    ensure_dir(args.cnhubert_dir, _CNHUBERT_HF_REPO)
    cnhubert.hubert_base_path = args.cnhubert_dir

    hubert_dir = os.path.join(args.opt_dir, "01_cnhubert")
    wav32dir = os.path.join(args.opt_dir, "02_wav32k")
    os.makedirs(args.opt_dir, exist_ok=True)
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    max_amplitude = 0.95
    blend_ratio = 0.5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = cnhubert.get_model()
    if is_half:
        model = model.half().to(device)
    else:
        model = model.to(device)

    nan_fails: list[tuple[str, str]] = []

    def _extract_hubert_and_wav32k(wav_name: str, wav_path: str) -> None:
        hubert_path = os.path.join(hubert_dir, f"{wav_name}.pt")
        audio = load_audio(wav_path, 32000)
        peak = np.abs(audio).max()
        if peak > 2.2:
            logger.warning("{}-filtered,{}", wav_name, peak)
            return
        audio_32k_int16 = (audio / peak * (max_amplitude * blend_ratio * 32768)) + ((1 - blend_ratio) * 32768) * audio
        audio_32k_float = (audio / peak * (max_amplitude * blend_ratio * 1145.14)) + ((1 - blend_ratio) * 1145.14) * audio
        audio_16k = librosa.resample(audio_32k_float, orig_sr=32000, target_sr=16000)
        tensor_wav16 = torch.from_numpy(audio_16k)
        if is_half:
            tensor_wav16 = tensor_wav16.half().to(device)
        else:
            tensor_wav16 = tensor_wav16.to(device)
        ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
        if np.isnan(ssl.detach().numpy()).sum() != 0:
            nan_fails.append((wav_name, wav_path))
            logger.warning("nan filtered:{}", wav_name)
            return
        wavfile.write(
            os.path.join(wav32dir, wav_name),
            32000,
            audio_32k_int16.astype("int16"),
        )
        _save_tensor(ssl, hubert_path, args.i_part)

    # 입력 파일 읽기 + 상태 필터링 + 파티셔닝
    with open(args.inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines = filter_label_lines(lines)

    for line in lines[args.i_part :: args.all_parts]:
        try:
            wav_name, spk_name, language, text = parse_label_line(line)
            wav_path = clean_path(wav_name)
            wav_name = os.path.basename(wav_path)
            _extract_hubert_and_wav32k(wav_name, wav_path)
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", line.split("|")[0], e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())

    # NaN 발생 시 FP32로 재시도
    if nan_fails and is_half:
        model = model.float()
        for wav_name, wav_path in nan_fails:
            try:
                _extract_hubert_and_wav32k(wav_name, wav_path)
            except Exception as e:
                logger.warning("{} -> FP32 재시도 실패 ({})", wav_name, e)
                logger.debug("상세 traceback:\n{}", traceback.format_exc())


if __name__ == "__main__":
    main()
