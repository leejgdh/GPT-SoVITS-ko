"""GPT-SoVITS TTS 엔트리포인트.

서브커맨드:
  serve     — REST API 서버 실행
  pipeline  — 전체 파이프라인 (Step 1~4) 일괄 실행
"""
from __future__ import annotations

import argparse
import glob
import logging as _logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# 공통: 로거 설정
# ---------------------------------------------------------------------------

_CONSOLE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss,SSS} [{extra[request_id]}] "
    "[<level>{level}</level>] {name}: {message}"
)
_FILE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss,SSS} [{extra[request_id]}] "
    "[{level}] {name}: {message}"
)


def _setup_logger(
    root_name: str,
    level: str = "INFO",
    log_dir: Path | None = None,
) -> None:
    logger.remove()

    def _patcher(record: dict) -> None:
        name = record["name"] or ""
        if name.startswith("src."):
            name = name[4:]
        if name == "__main__":
            record["name"] = root_name
        elif not name.startswith(f"{root_name}."):
            record["name"] = f"{root_name}.{name}"

    logger.configure(extra={"request_id": "-"}, patcher=_patcher)
    logger.add(
        sys.stderr, format=_CONSOLE_FMT,
        level=level.upper(), colorize=True,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_dir / f"{root_name}.log"),
            format=_FILE_FMT, level="DEBUG",
            rotation="10 MB", retention=5, encoding="utf-8",
        )

    class _InterceptHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            try:
                lvl: str | int = logger.level(record.levelname).name
            except ValueError:
                lvl = record.levelno
            frame, depth = _logging.currentframe(), 2
            while frame and frame.f_code.co_filename == _logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                lvl, record.getMessage(),
            )

    _logging.basicConfig(
        handlers=[_InterceptHandler()], level=0, force=True,
    )


# ---------------------------------------------------------------------------
# serve 커맨드
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> None:
    """REST API 서버를 실행한다."""
    import uvicorn

    from src.config.config import Config, load_config

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        os.environ["TTS_SERVICE_CONFIG"] = str(config_path)
    else:
        logger.info("설정 파일 없음 — 기본값으로 실행합니다")
        config = Config()

    log_level = "DEBUG" if args.verbose else config.log_level
    _setup_logger(
        "tts_service",
        level=log_level,
        log_dir=Path(__file__).resolve().parent / "logs",
    )

    host = args.host or config.service.host
    port = args.port or config.service.port

    logger.info("GPT-SoVITS TTS Service 시작: {}:{}", host, port)

    uvicorn.run(
        "src.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False,
    )


# ---------------------------------------------------------------------------
# pipeline 커맨드
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parent)

_SOVITS_TRAIN_SCRIPT = {
    "v2": "scripts/training/s2_train_vits.py",
    "v2Pro": "scripts/training/s2_train_vits.py",
    "v2ProPlus": "scripts/training/s2_train_vits.py",
    "v3": "scripts/training/s2_train_cfm.py",
    "v4": "scripts/training/s2_train_cfm.py",
}

_SV_VERSIONS = {"v2Pro", "v2ProPlus"}


def _run(cmd: list[str], label: str) -> None:
    """서브프로세스를 실행하고, 실패 시 즉시 종료한다."""
    logger.info(">>> {} 시작", label)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("<<< {} 실패 (exit={})", label, result.returncode)
        sys.exit(result.returncode)
    logger.info("<<< {} 완료 ({:.0f}초)", label, elapsed)


def _find_latest(directory: str, pattern: str) -> str | None:
    """디렉토리에서 가장 최근 수정된 파일을 찾는다."""
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None


def _find_ref_audio(voice_dir: str) -> str | None:
    """step1/03_vocal 에서 첫 번째 오디오 파일을 찾는다."""
    vocal_dir = os.path.join(voice_dir, "step1", "03_vocal")
    if not os.path.isdir(vocal_dir):
        return None
    for f in sorted(os.listdir(vocal_dir)):
        if f.endswith((".wav", ".flac", ".mp3")):
            return os.path.join(vocal_dir, f)
    return None


def _find_ref_text(voice_dir: str, ref_audio: str) -> str | None:
    """ASR 라벨에서 ref_audio에 해당하는 텍스트를 찾는다."""
    asr_dir = os.path.join(voice_dir, "step1", "04_asr")
    if not os.path.isdir(asr_dir):
        return None
    for list_file in glob.glob(os.path.join(asr_dir, "*.list")):
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if (
                    len(parts) >= 4
                    and os.path.basename(parts[0]) == os.path.basename(ref_audio)
                ):
                    return parts[3]
    return None


def _save_voice_yaml(
    voice_dir: str, version: str,
    ref_audio: str, ref_text: str, ref_lang: str,
) -> None:
    """step4 완료 후 voice.yaml을 자동 생성한다."""
    from src.config.voice import save_voice_yaml

    step3 = os.path.join(voice_dir, "step3", version)
    gpt_weights = _find_latest(
        os.path.join(step3, "02_gpt_weights"), "*.ckpt",
    )
    sovits_weights = _find_latest(
        os.path.join(step3, "04_sovits_weights"), "*.pth",
    )

    if gpt_weights is None or sovits_weights is None:
        logger.warning(
            "가중치를 찾을 수 없어 voice.yaml을 생성하지 않습니다",
        )
        return

    save_voice_yaml(
        voice_dir,
        name=os.path.basename(os.path.abspath(voice_dir)),
        version=version,
        ref_audio=ref_audio,
        ref_text=ref_text,
        ref_lang=ref_lang,
        gpt_weights=gpt_weights,
        sovits_weights=sovits_weights,
    )


def _clean_step_dir(voice_dir: str, step: str, version: str, label: str) -> None:
    """step 출력 디렉토리를 삭제한다."""
    step_dir = os.path.join(voice_dir, step, version)
    if os.path.exists(step_dir):
        logger.info("기존 {} 결과 삭제: {}", label, step_dir)
        shutil.rmtree(step_dir)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """전체 파이프라인 (Step 1~4)을 순서대로 실행한다."""
    _setup_logger("pipeline", level="DEBUG" if args.verbose else "INFO")

    voice_dir = args.voice_dir
    version = args.version
    skip = set(args.skip or [])
    py = sys.executable

    total_start = time.time()
    logger.info("=== 파이프라인 시작: {} (version={}) ===", voice_dir, version)

    # -- Step 1: 데이터 준비 --
    if "step1" not in skip:
        _run([py, "scripts/data_preparation/denoise.py", "--voice-dir", voice_dir],
             "Step1-1 denoise")
        _run([py, "scripts/data_preparation/slice_audio.py", "--voice-dir", voice_dir],
             "Step1-2 slice_audio")
        _run(
            [py, "scripts/data_preparation/uvr5_separate.py",
             "--voice-dir", voice_dir],
            "Step1-3 uvr5_separate",
        )
        _run([py, "scripts/data_preparation/asr_whisper.py", "--voice-dir", voice_dir],
             "Step1-4 asr_whisper")

    # -- Step 2: 전처리 --
    if "step2" not in skip:
        _clean_step_dir(voice_dir, "step2", version, "전처리")
        _run([py, "scripts/preprocessing/1-get-text.py",
              "--voice-dir", voice_dir, "--version", version],
             "Step2-1 get-text")
        _run([py, "scripts/preprocessing/2-get-hubert-wav32k.py",
              "--voice-dir", voice_dir, "--version", version],
             "Step2-2 get-hubert-wav32k")
        if version in _SV_VERSIONS:
            _run([py, "scripts/preprocessing/2-get-sv.py",
                  "--voice-dir", voice_dir, "--version", version],
                 "Step2-SV get-sv")
        _run([py, "scripts/preprocessing/3-get-semantic.py",
              "--voice-dir", voice_dir, "--version", version],
             "Step2-3 get-semantic")

    # -- Step 3: 학습 --
    if "step3" not in skip:
        _clean_step_dir(voice_dir, "step3", version, "학습")
        gpt_cmd = [py, "scripts/training/s1_train.py",
                   "--voice-dir", voice_dir, "--version", version]
        if args.epochs is not None:
            gpt_cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            gpt_cmd += ["--batch-size", str(args.batch_size)]
        _run(gpt_cmd, "Step3-1 GPT train")

        sovits_script = _SOVITS_TRAIN_SCRIPT[version]
        sovits_cmd = [py, sovits_script,
                      "--voice-dir", voice_dir, "--version", version]
        if args.epochs is not None:
            sovits_cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            sovits_cmd += ["--batch-size", str(args.batch_size)]
        _run(sovits_cmd, "Step3-2 SoVITS train")

    # -- Step 4: 추론 --
    if "step4" not in skip:
        _clean_step_dir(voice_dir, "step4", version, "추론")
        ref_audio = args.ref_audio or _find_ref_audio(voice_dir)
        if ref_audio is None:
            logger.error("참조 오디오를 찾을 수 없습니다. --ref-audio를 지정하세요.")
            sys.exit(1)

        ref_text = args.ref_text or _find_ref_text(voice_dir, ref_audio)
        if ref_text is None:
            logger.error("참조 텍스트를 찾을 수 없습니다. --ref-text를 지정하세요.")
            sys.exit(1)

        _run([py, "scripts/inference/inference_cli.py",
              "--voice-dir", voice_dir, "--version", version,
              "--ref-audio", ref_audio,
              "--ref-text", ref_text,
              "--text", args.output_text],
             "Step4 inference")

        # voice.yaml 자동 생성
        _save_voice_yaml(voice_dir, version, ref_audio, ref_text, args.ref_lang)

    total_elapsed = time.time() - total_start
    logger.info("=== 파이프라인 완료 ({:.0f}초) ===", total_elapsed)


# ---------------------------------------------------------------------------
# 메인 파서
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS TTS Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- serve ---
    sp_serve = sub.add_parser("serve", help="REST API 서버 실행")
    sp_serve.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    sp_serve.add_argument(
        "-c", "--config", default="conf.yaml",
        help="설정 파일 경로",
    )
    sp_serve.add_argument(
        "--host", default=None,
        help="서버 호스트 (설정 파일보다 우선)",
    )
    sp_serve.add_argument(
        "--port", type=int, default=None,
        help="서버 포트 (설정 파일보다 우선)",
    )

    # --- pipeline ---
    sp_pipe = sub.add_parser(
        "pipeline", help="전체 파이프라인 (Step 1~4) 일괄 실행",
    )
    sp_pipe.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    sp_pipe.add_argument(
        "--voice-dir", required=True,
        help="캐릭터 음성 폴더 (예: data/voice/lunabi)",
    )
    sp_pipe.add_argument(
        "--output-text", required=True,
        help="Step 4에서 합성할 텍스트",
    )
    sp_pipe.add_argument(
        "--version", default="v2Pro",
        choices=["v2", "v3", "v4", "v2Pro", "v2ProPlus"],
        help="모델 버전 (기본: v2Pro)",
    )
    sp_pipe.add_argument(
        "--epochs", type=int, default=None,
        help="학습 에포크 수",
    )
    sp_pipe.add_argument(
        "--batch-size", type=int, default=None,
        help="배치 크기",
    )
    sp_pipe.add_argument(
        "--ref-audio", default=None,
        help="Step 4 참조 오디오 (미지정 시 자동 탐색)",
    )
    sp_pipe.add_argument(
        "--ref-text", default=None,
        help="Step 4 참조 텍스트 (미지정 시 ASR 라벨에서 자동)",
    )
    sp_pipe.add_argument(
        "--ref-lang", default="ko",
        choices=["ko", "en", "ja"],
        help="참조 언어 (기본: ko)",
    )
    sp_pipe.add_argument(
        "--skip", nargs="*", default=[], metavar="STEP",
        choices=["step1", "step2", "step3", "step4"],
        help="건너뛸 스텝 (예: --skip step1 step2)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "pipeline":
        _cmd_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
