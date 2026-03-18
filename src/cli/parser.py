"""CLI argparse 빌더."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS TTS Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # 공통 인자 헬퍼
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
        sp.add_argument("-c", "--config", default="conf.yaml", help="설정 파일 경로")

    def _add_voice_dir(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--voice-dir", required=True, help="캐릭터 음성 폴더")

    def _add_version(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--version", default="v2Pro",
            choices=["v2", "v3", "v4", "v2Pro", "v2ProPlus"],
            help="모델 버전 (기본: v2Pro)",
        )

    # ── serve ──
    sp_serve = sub.add_parser("serve", help="REST API 서버 실행")
    _add_common(sp_serve)
    sp_serve.add_argument("--host", default=None, help="서버 호스트")
    sp_serve.add_argument("--port", type=int, default=None, help="서버 포트")

    # ── pipeline ──
    sp_pipe = sub.add_parser("pipeline", help="전체 파이프라인 (Step 1~4) 일괄 실행")
    _add_common(sp_pipe)
    _add_voice_dir(sp_pipe)
    _add_version(sp_pipe)
    sp_pipe.add_argument("--output-text", required=True, help="Step 4에서 합성할 텍스트")
    sp_pipe.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    sp_pipe.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    sp_pipe.add_argument("--ref-audio", default=None, help="Step 4 참조 오디오")
    sp_pipe.add_argument("--ref-text", default=None, help="Step 4 참조 텍스트")
    sp_pipe.add_argument("--ref-lang", default="ko", choices=["ko", "en", "ja"])
    sp_pipe.add_argument(
        "--skip", nargs="*", default=[], metavar="STEP",
        choices=["step1", "step2", "step3", "step4"],
        help="건너뛸 스텝 (예: --skip step1 step2)",
    )

    # ── step1 (순차 실행) ──
    sp_s1 = sub.add_parser("step1", help="데이터 준비 (denoise → slice → UVR5 → ASR)")
    _add_common(sp_s1)
    _add_voice_dir(sp_s1)

    # ── step1 하위 커맨드 ──
    sp = sub.add_parser("denoise", help="Step1-1: FRCRN 노이즈 제거")
    _add_common(sp)
    _add_voice_dir(sp)

    sp = sub.add_parser("slice", help="Step1-2: 무음 기반 슬라이싱")
    _add_common(sp)
    _add_voice_dir(sp)

    sp = sub.add_parser("uvr5", help="Step1-3: UVR5 보컬 분리")
    _add_common(sp)
    _add_voice_dir(sp)

    sp = sub.add_parser("asr", help="Step1-4: Whisper ASR")
    _add_common(sp)
    _add_voice_dir(sp)

    sp = sub.add_parser("classify", help="ASR 라벨의 pending 상태를 Voice Checker CNN으로 재분류")
    _add_common(sp)
    _add_voice_dir(sp)

    # ── step2 (순차 실행) ──
    sp_s2 = sub.add_parser("step2", help="전처리 (text → hubert → semantic)")
    _add_common(sp_s2)
    _add_voice_dir(sp_s2)
    _add_version(sp_s2)

    # ── step2 하위 커맨드 ──
    sp = sub.add_parser("get-text", help="Step2-1: 음소 추출")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)

    sp = sub.add_parser("get-hubert", help="Step2-2: HuBERT + wav32k")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)

    sp = sub.add_parser("get-sv", help="Step2-SV: 화자 임베딩 (v2Pro/v2ProPlus)")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)

    sp = sub.add_parser("get-semantic", help="Step2-3: Semantic 토큰 추출")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)

    # ── step3 (순차 실행) ──
    sp_s3 = sub.add_parser("step3", help="학습 (GPT AR + SoVITS)")
    _add_common(sp_s3)
    _add_voice_dir(sp_s3)
    _add_version(sp_s3)
    sp_s3.add_argument("--epochs", type=int, default=None)
    sp_s3.add_argument("--batch-size", type=int, default=None)

    # ── step3 하위 커맨드 ──
    sp = sub.add_parser("train-gpt", help="Step3-1: GPT AR 모델 학습")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)

    sp = sub.add_parser("train-sovits", help="Step3-2: SoVITS 모델 학습")
    _add_common(sp)
    _add_voice_dir(sp)
    _add_version(sp)
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)

    # ── step4 ──
    sp_s4 = sub.add_parser("step4", help="추론 + voice.yaml 자동 생성")
    _add_common(sp_s4)
    _add_voice_dir(sp_s4)
    _add_version(sp_s4)
    sp_s4.add_argument("--output-text", required=True, help="합성할 텍스트")
    sp_s4.add_argument("--ref-audio", default=None)
    sp_s4.add_argument("--ref-text", default=None)
    sp_s4.add_argument("--ref-lang", default="ko", choices=["ko", "en", "ja"])

    return parser
