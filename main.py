"""GPT-SoVITS TTS 엔트리포인트.

서브커맨드:
  serve     — REST API 서버 실행
  pipeline  — 전체 파이프라인 (Step 1~4) 일괄 실행
  step1     — 데이터 준비 (denoise → slice → UVR5 → ASR)
  step2     — 전처리 (text → hubert → semantic)
  step3     — 학습 (GPT AR + SoVITS)
  step4     — 추론 + voice.yaml 자동 생성
"""
from __future__ import annotations

import sys

from src.cli.parser import build_parser
from src.cli.pipeline import (
    cmd_pipeline,
    cmd_step1,
    cmd_step2,
    cmd_step3,
    cmd_step4,
)
from src.cli.serve import cmd_serve
from src.cli.server import start_server_background, wait_for_server

_CMD_MAP = {
    "serve": cmd_serve,
    "pipeline": cmd_pipeline,
    "step1": cmd_step1,
    "step2": cmd_step2,
    "step3": cmd_step3,
    "step4": cmd_step4,
}

_SERVER_COMMANDS = {"step1", "step2", "step3", "step4", "pipeline"}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        args.command = "serve"
        # serve 기본 인자 설정
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
