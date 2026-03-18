"""GPT-SoVITS TTS 엔트리포인트.

서브커맨드:
  serve       — REST API 서버 실행 (기본)
  pipeline    — 전체 파이프라인 (Step 1~4) 일괄 실행

  step1       — 데이터 준비 (denoise → slice → UVR5 → ASR)
  step2       — 전처리 (text → hubert → semantic)
  step3       — 학습 (GPT AR + SoVITS)
  step4       — 추론 + voice.yaml 자동 생성

  denoise     — Step1-1: FRCRN 노이즈 제거
  slice       — Step1-2: 무음 기반 슬라이싱
  uvr5        — Step1-3: UVR5 보컬 분리
  asr         — Step1-4: Whisper ASR
  classify    — ASR 라벨 pending → CNN 재분류

  get-text    — Step2-1: 음소 추출
  get-hubert  — Step2-2: HuBERT + wav32k
  get-sv      — Step2-SV: 화자 임베딩
  get-semantic— Step2-3: Semantic 토큰

  train-gpt   — Step3-1: GPT AR 학습
  train-sovits— Step3-2: SoVITS 학습
"""
from __future__ import annotations

import sys

from src.cli.parser import build_parser
from src.cli.pipeline import (
    cmd_asr,
    cmd_classify,
    cmd_denoise,
    cmd_get_hubert,
    cmd_get_semantic,
    cmd_get_sv,
    cmd_get_text,
    cmd_pipeline,
    cmd_slice,
    cmd_step1,
    cmd_step2,
    cmd_step3,
    cmd_step4,
    cmd_train_gpt,
    cmd_train_sovits,
    cmd_uvr5,
)
from src.cli.serve import cmd_serve
from src.cli.server import start_server_background, wait_for_server

_CMD_MAP = {
    "serve": cmd_serve,
    "pipeline": cmd_pipeline,
    # step 묶음
    "step1": cmd_step1,
    "step2": cmd_step2,
    "step3": cmd_step3,
    "step4": cmd_step4,
    # step1 하위
    "denoise": cmd_denoise,
    "slice": cmd_slice,
    "uvr5": cmd_uvr5,
    "asr": cmd_asr,
    "classify": cmd_classify,
    # step2 하위
    "get-text": cmd_get_text,
    "get-hubert": cmd_get_hubert,
    "get-sv": cmd_get_sv,
    "get-semantic": cmd_get_semantic,
    # step3 하위
    "train-gpt": cmd_train_gpt,
    "train-sovits": cmd_train_sovits,
}

_SERVER_COMMANDS = {
    "step1", "step2", "step3", "step4", "pipeline",
    "denoise", "slice", "uvr5", "asr", "classify",
    "get-text", "get-hubert", "get-sv", "get-semantic",
    "train-gpt", "train-sovits",
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        args.command = "serve"
        for attr, default in [("verbose", False), ("config", "conf.yaml"), ("host", None), ("port", None)]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    handler = _CMD_MAP.get(args.command)

    if args.command in _SERVER_COMMANDS:
        config_path = getattr(args, "config", "conf.yaml")
        start_server_background(config_path)

    handler(args)

    if args.command in _SERVER_COMMANDS:
        wait_for_server()


if __name__ == "__main__":
    main()
