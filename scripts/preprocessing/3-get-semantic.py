# -*- coding: utf-8 -*-
"""시맨틱 토큰 추출 (전처리 3단계).

HuBERT 임베딩에서 SoVITS VQ-VAE를 통해
시맨틱 토큰(양자화 코드)을 추출한다.

출력:
  - {opt_dir}/name2semantic-{i_part}.tsv  (시맨틱 토큰)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback

from loguru import logger

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import filter_label_lines, parse_label_line, setup_paths

setup_paths()
# ---------------------------------------------------------------

import torch
import utils

from tools.dl_utils import ensure_file
from tools.my_utils import clean_path

logging.getLogger("numba").setLevel(logging.WARNING)

_DEFAULT_PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
_DEFAULT_S2CONFIG = "GPT_SoVITS/configs/s2.json"
_HF_REPO = "lj1995/GPT-SoVITS"


def _detect_version(pretrained_path: str) -> str:
    """SoVITS 체크포인트 파일 크기로 모델 버전을 감지한다."""
    size = os.path.getsize(pretrained_path)
    if size < 82978 * 1024:
        return "v1"
    elif size < 100 * 1024 * 1024:
        return "v2"
    elif size < 103520 * 1024:
        return "v1"
    elif size < 700 * 1024 * 1024:
        return "v2"
    else:
        return "v3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="시맨틱 토큰 추출 (VQ 양자화)")
    parser.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--inp-text", default=None, help="입력 텍스트 파일 (기본: {voice-dir}/step1/04_asr/03_vocal.list)")
    parser.add_argument("--opt-dir", default=None, help="출력 디렉토리 (기본: {voice-dir}/step2)")
    parser.add_argument(
        "--pretrained-s2g", default=_DEFAULT_PRETRAINED_S2G,
        help=f"SoVITS pretrained 모델 경로 (기본: {_DEFAULT_PRETRAINED_S2G})",
    )
    parser.add_argument(
        "--s2config-path", default=_DEFAULT_S2CONFIG,
        help=f"SoVITS 설정 파일 경로 (기본: {_DEFAULT_S2CONFIG})",
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

    # SoVITS pretrained 모델 자동 다운로드
    ensure_file(
        args.pretrained_s2g,
        _HF_REPO,
        filename=os.path.basename(args.pretrained_s2g),
        subfolder="gsv-v2final-pretrained",
    )

    version = _detect_version(args.pretrained_s2g)
    is_half = (not args.no_half) and torch.cuda.is_available()

    if version != "v3":
        from module.models import SynthesizerTrn
    else:
        from module.models import SynthesizerTrnV3 as SynthesizerTrn

    hubert_dir = os.path.join(args.opt_dir, "01_cnhubert")
    semantic_path = os.path.join(args.opt_dir, f"name2semantic-{args.i_part}.tsv")

    os.makedirs(args.opt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hps = utils.get_hparams_from_file(args.s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version=version,
        **hps.model,
    )
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    logger.info(
        "vq_model loaded: {}",
        vq_model.load_state_dict(
            torch.load(args.pretrained_s2g, map_location="cpu", weights_only=False)["weight"],
            strict=False,
        ),
    )

    def name2go(wav_name: str, lines: list) -> None:
        hubert_path = os.path.join(hubert_dir, f"{wav_name}.pt")
        if not os.path.exists(hubert_path):
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    # 입력 파일 읽기 + 상태 필터링 + 파티셔닝
    with open(args.inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines = filter_label_lines(lines)

    lines1: list = []
    for line in lines[args.i_part :: args.all_parts]:
        try:
            wav_name, spk_name, language, text = parse_label_line(line)
            wav_name = clean_path(wav_name)
            wav_name = os.path.basename(wav_name)
            name2go(wav_name, lines1)
        except Exception as e:
            logger.warning("{} -> 건너뜀 ({})", line.split("|")[0], e)
            logger.debug("상세 traceback:\n{}", traceback.format_exc())

    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))


if __name__ == "__main__":
    main()
