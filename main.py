"""GPT-SoVITS TTS 엔트리포인트.

서브커맨드:
  serve         — REST API 서버 실행 (기본, 추론/학습 양쪽 이미지에서 동작)
  cleanup-voice — 학습 산출물 정리 (voice.yaml 의 weight/ref_audio 만 보존)

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

이미지 모드:
  TTS_MODE=infer 환경변수가 설정된 추론 이미지에서는 학습 명령을 거부한다.
  학습 이미지는 모든 명령 사용 가능.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 호스트 실행 시 이 프로젝트 루트의 .env.local (gitignored) 을 dotenv 로 로드한다.
# 현재는 참조하는 env 키가 없어도 구조 통일 차원에서 슬롯 유지.
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env.local", override=True)

from src.cli.parser import build_parser

# 추론 이미지에서도 동작하는 명령
_INFER_COMMANDS = {"serve", "cleanup-voice"}

# 학습 이미지에서만 동작하는 명령 (training extras 필요).
# 이 명령들은 실행 시 백그라운드 서버 (라벨 검수 UI / 학습 모니터링) 도 함께 띄운다.
_TRAIN_COMMANDS = {
    "pipeline",
    "step1", "step2", "step3", "step4",
    "denoise", "slice", "uvr5", "asr", "classify",
    "get-text", "get-hubert", "get-sv", "get-semantic",
    "train-gpt", "train-sovits",
}


def _is_infer_only() -> bool:
    return os.environ.get("TTS_MODE", "").lower() == "infer"


def _load_handler(command: str):
    """명령에 대응하는 handler 를 lazy 로 로드한다.

    추론 모드 (TTS_MODE=infer) 에서 학습 명령을 호출하면 즉시 안내 후 종료한다 —
    이미지 안에 학습 의존성(faster-whisper / peft / funasr 등)이 없어 어차피
    subprocess 단계에서 ModuleNotFoundError 가 나기 때문에 사전 차단.
    """
    if command in _INFER_COMMANDS:
        if command == "serve":
            from src.cli.serve import cmd_serve
            return cmd_serve
        if command == "cleanup-voice":
            from src.cli.pipeline import cmd_cleanup_voice
            return cmd_cleanup_voice

    if command in _TRAIN_COMMANDS:
        if _is_infer_only():
            sys.stderr.write(
                f"명령 '{command}' 은 학습 이미지에서만 사용 가능합니다.\n"
                "현재 이미지는 추론 전용 (TTS_MODE=infer) — training extras 가 없습니다.\n"
                "학습 이미지 빌드: docker build --target train -t tts:train .\n"
            )
            sys.exit(2)
        from src.cli import pipeline as p
        return {
            "pipeline": p.cmd_pipeline,
            "step1": p.cmd_step1,
            "step2": p.cmd_step2,
            "step3": p.cmd_step3,
            "step4": p.cmd_step4,
            "denoise": p.cmd_denoise,
            "slice": p.cmd_slice,
            "uvr5": p.cmd_uvr5,
            "asr": p.cmd_asr,
            "classify": p.cmd_classify,
            "get-text": p.cmd_get_text,
            "get-hubert": p.cmd_get_hubert,
            "get-sv": p.cmd_get_sv,
            "get-semantic": p.cmd_get_semantic,
            "train-gpt": p.cmd_train_gpt,
            "train-sovits": p.cmd_train_sovits,
        }[command]

    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        args.command = "serve"
        for attr, default in [
            ("verbose", False), ("config", "config.yaml"),
            ("host", None), ("port", None),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    handler = _load_handler(args.command)
    if handler is None:
        sys.stderr.write(f"알 수 없는 명령: {args.command}\n")
        sys.exit(2)

    if args.command in _TRAIN_COMMANDS:
        from src.cli.server import start_server_background, wait_for_server
        config_path = getattr(args, "config", "config.yaml")
        start_server_background(config_path)
        handler(args)
        wait_for_server()
    else:
        handler(args)


if __name__ == "__main__":
    main()
