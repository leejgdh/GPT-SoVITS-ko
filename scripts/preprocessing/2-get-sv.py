# -*- coding: utf-8 -*-
"""화자 임베딩 추출 (전처리 2단계-b).

ERes2NetV2 모델을 사용하여 32kHz 오디오에서
화자 임베딩(20480차원)을 추출한다.

출력:
  - {opt_dir}/04_sv/{wav_name}.pt  (화자 임베딩)
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

import kaldi as Kaldi
import torch
import torchaudio
from ERes2NetV2 import ERes2NetV2

from tools.utils.audio import clean_path

_DEFAULT_SV_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


def _save_tensor(tensor: torch.Tensor, path: str, i_part: int) -> None:
    """torch.save가 비ASCII 경로를 지원하지 않는 문제 우회."""
    dir_ = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth" % (time(), i_part)
    torch.save(tensor, tmp_path)
    shutil.move(tmp_path, os.path.join(dir_, name))


class SpeakerEmbeddingExtractor:
    """ERes2NetV2 기반 화자 임베딩 추출기."""

    def __init__(self, sv_model_path: str, device: str, is_half: bool) -> None:
        pretrained_state = torch.load(sv_model_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.resampler = torchaudio.transforms.Resample(32000, 16000).to(device)
        if is_half:
            self.embedding_model = self.embedding_model.half().to(device)
        else:
            self.embedding_model = self.embedding_model.to(device)
        self.is_half = is_half

    def compute_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """(1, x) 형태의 32kHz 오디오에서 화자 임베딩을 추출한다."""
        with torch.no_grad():
            wav = self.resampler(wav)
            if self.is_half:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="화자 임베딩 추출 (ERes2NetV2)")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--inp-text", default=None, help="입력 텍스트 파일 (기본: {voice-dir}/step1/04_asr/03_vocal.list)")
    parser.add_argument("--opt-dir", default=None, help="출력 디렉토리 (기본: {voice-dir}/step2)")
    parser.add_argument(
        "--sv-path", default=_DEFAULT_SV_PATH,
        help=f"ERes2Net 모델 체크포인트 경로 (기본: {_DEFAULT_SV_PATH})",
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sv_dir = os.path.join(args.opt_dir, "04_sv")
    wav32dir = os.path.join(args.opt_dir, "02_wav32k")
    os.makedirs(args.opt_dir, exist_ok=True)
    os.makedirs(sv_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    if not os.path.exists(args.sv_path):
        from tools.utils.download import ensure_file
        logger.info("ERes2NetV2 모델 다운로드 중...")
        ensure_file(
            local_path=args.sv_path,
            repo_id="lj1995/GPT-SoVITS",
            filename="sv/pretrained_eres2netv2w24s4ep4.ckpt",
        )
        if not os.path.exists(args.sv_path):
            raise FileNotFoundError(f"ERes2NetV2 모델 다운로드 실패: {args.sv_path}")

    extractor = SpeakerEmbeddingExtractor(args.sv_path, device, is_half)

    def _extract_embedding(wav_name: str, wav_path: str) -> None:
        sv_path = os.path.join(sv_dir, f"{wav_name}.pt")
        wav_path = os.path.join(wav32dir, wav_name)
        wav32k, sr0 = torchaudio.load(wav_path)
        assert sr0 == 32000
        wav32k = wav32k.to(device)
        emb = extractor.compute_embedding(wav32k).cpu()
        _save_tensor(emb, sv_path, args.i_part)

    # 입력 파일 읽기 + 상태 필터링 + 파티셔닝
    with open(args.inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines = filter_label_lines(lines)

    for line in lines[args.i_part :: args.all_parts]:
        try:
            wav_name, spk_name, language, text = parse_label_line(line)
            wav_path = clean_path(wav_name)
            wav_name = os.path.basename(wav_path)
            _extract_embedding(wav_name, wav_path)
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", line.split("|")[0], e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())


if __name__ == "__main__":
    main()
