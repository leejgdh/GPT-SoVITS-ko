"""파이프라인 step 실행 로직 + step/pipeline 커맨드."""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from src.cli.logger import LOG_DIR, setup_logger

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

_SOVITS_TRAIN_SCRIPT = {
    "v2": "scripts/training/s2_train_vits.py",
    "v2Pro": "scripts/training/s2_train_vits.py",
    "v2ProPlus": "scripts/training/s2_train_vits.py",
    "v3": "scripts/training/s2_train_cfm.py",
    "v4": "scripts/training/s2_train_cfm.py",
}

_SV_VERSIONS = {"v2Pro", "v2ProPlus"}

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


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
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None


def _find_ref_audio(voice_dir: str) -> str | None:
    vocal_dir = os.path.join(voice_dir, "step1", "03_vocal")
    if not os.path.isdir(vocal_dir):
        return None
    for f in sorted(os.listdir(vocal_dir)):
        if f.endswith((".wav", ".flac", ".mp3")):
            return os.path.join(vocal_dir, f)
    return None


def _find_ref_text(voice_dir: str, ref_audio: str) -> str | None:
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
    from src.config.voice import save_voice_yaml

    step3 = os.path.join(voice_dir, "step3", version)
    gpt_weights = _find_latest(os.path.join(step3, "02_gpt_weights"), "*.ckpt")
    sovits_weights = _find_latest(os.path.join(step3, "04_sovits_weights"), "*.pth")

    if gpt_weights is None or sovits_weights is None:
        logger.warning("가중치를 찾을 수 없어 voice.yaml을 생성하지 않습니다")
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
    step_dir = os.path.join(voice_dir, step, version)
    if os.path.exists(step_dir):
        logger.info("기존 {} 결과 삭제: {}", label, step_dir)
        shutil.rmtree(step_dir)


def _load_voice_checker_model(config_path: str = "conf.yaml") -> str | None:
    from src.config.config import Config, load_config

    path = Path(config_path)
    if not path.exists():
        return None
    config = load_config(path)
    if config.voice_checker is None:
        return None
    model_path = config.voice_checker.inference.model_path
    if Path(model_path).exists():
        return model_path
    logger.warning("voice_checker 모델 경로가 존재하지 않습니다: {}", model_path)
    return None


# ---------------------------------------------------------------------------
# step 실행 함수
# ---------------------------------------------------------------------------


def _ensure_voice_yaml(voice_dir: str) -> None:
    """voice.yaml이 없으면 초기 파일을 생성한다 (available: false)."""
    yaml_path = os.path.join(voice_dir, "voice.yaml")
    if os.path.exists(yaml_path):
        return
    import yaml
    name = os.path.basename(os.path.abspath(voice_dir))
    data = {"name": name, "available": False}
    os.makedirs(voice_dir, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info("voice.yaml 초기 생성: {} (available: false)", yaml_path)


def run_step1(voice_dir: str, config_path: str = "conf.yaml") -> None:
    _ensure_voice_yaml(voice_dir)
    py = sys.executable
    _run([py, "scripts/data_preparation/denoise.py", "--voice-dir", voice_dir],
         "Step1-1 denoise")
    _run([py, "scripts/data_preparation/slice_audio.py", "--voice-dir", voice_dir],
         "Step1-2 slice_audio")
    _run([py, "scripts/data_preparation/uvr5_separate.py", "--voice-dir", voice_dir],
         "Step1-3 uvr5_separate")

    asr_cmd = [py, "scripts/data_preparation/asr_whisper.py", "--voice-dir", voice_dir]
    vc_model = _load_voice_checker_model(config_path)
    if vc_model:
        asr_cmd += ["--voice-checker-model", vc_model]
        logger.info("Voice Checker 활성화: {}", vc_model)
    _run(asr_cmd, "Step1-4 asr_whisper")


def run_step2(voice_dir: str, version: str) -> None:
    py = sys.executable
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


def run_step3(
    voice_dir: str, version: str,
    epochs: int | None = None, batch_size: int | None = None,
) -> None:
    py = sys.executable
    _clean_step_dir(voice_dir, "step3", version, "학습")

    gpt_cmd = [py, "scripts/training/s1_train.py",
               "--voice-dir", voice_dir, "--version", version]
    if epochs is not None:
        gpt_cmd += ["--epochs", str(epochs)]
    if batch_size is not None:
        gpt_cmd += ["--batch-size", str(batch_size)]
    _run(gpt_cmd, "Step3-1 GPT train")

    sovits_script = _SOVITS_TRAIN_SCRIPT[version]
    sovits_cmd = [py, sovits_script,
                  "--voice-dir", voice_dir, "--version", version]
    if epochs is not None:
        sovits_cmd += ["--epochs", str(epochs)]
    if batch_size is not None:
        sovits_cmd += ["--batch-size", str(batch_size)]
    _run(sovits_cmd, "Step3-2 SoVITS train")


def run_step4(
    voice_dir: str, version: str, output_text: str,
    ref_audio: str | None = None, ref_text: str | None = None,
    ref_lang: str = "ko",
) -> None:
    py = sys.executable
    _clean_step_dir(voice_dir, "step4", version, "추론")

    ref_audio = ref_audio or _find_ref_audio(voice_dir)
    if ref_audio is None:
        logger.error("참조 오디오를 찾을 수 없습니다. --ref-audio를 지정하세요.")
        sys.exit(1)

    ref_text = ref_text or _find_ref_text(voice_dir, ref_audio)
    if ref_text is None:
        logger.error("참조 텍스트를 찾을 수 없습니다. --ref-text를 지정하세요.")
        sys.exit(1)

    _run([py, "scripts/inference/inference_cli.py",
          "--voice-dir", voice_dir, "--version", version,
          "--ref-audio", ref_audio,
          "--ref-text", ref_text,
          "--text", output_text],
         "Step4 inference")

    _save_voice_yaml(voice_dir, version, ref_audio, ref_text, ref_lang)


# ---------------------------------------------------------------------------
# CLI 커맨드
# ---------------------------------------------------------------------------


def cmd_step1(args: argparse.Namespace) -> None:
    setup_logger("step1", level="DEBUG" if args.verbose else "INFO")
    logger.info("=== Step 1 시작: {} ===", args.voice_dir)
    t0 = time.time()
    run_step1(args.voice_dir, args.config)
    logger.info("=== Step 1 완료 ({:.0f}초) ===", time.time() - t0)
    logger.info(
        "[다음 단계]\n"
        "  (선택) ASR 라벨 검수: http://localhost:9880/review\n"
        "  바로 진행: python main.py step2 --voice-dir {}",
        args.voice_dir,
    )


def cmd_step2(args: argparse.Namespace) -> None:
    setup_logger("step2", level="DEBUG" if args.verbose else "INFO")
    logger.info("=== Step 2 시작: {} (version={}) ===", args.voice_dir, args.version)
    t0 = time.time()
    run_step2(args.voice_dir, args.version)
    logger.info("=== Step 2 완료 ({:.0f}초) ===", time.time() - t0)
    logger.info(
        "[다음 단계] python main.py step3 --voice-dir {} --version {}",
        args.voice_dir, args.version,
    )


def cmd_step3(args: argparse.Namespace) -> None:
    setup_logger("step3", level="DEBUG" if args.verbose else "INFO")
    logger.info("=== Step 3 시작: {} (version={}) ===", args.voice_dir, args.version)
    t0 = time.time()
    run_step3(args.voice_dir, args.version, args.epochs, args.batch_size)
    logger.info("=== Step 3 완료 ({:.0f}초) ===", time.time() - t0)
    logger.info(
        "[다음 단계] python main.py step4 --voice-dir {} --version {} --output-text '합성할 텍스트'",
        args.voice_dir, args.version,
    )


def cmd_step4(args: argparse.Namespace) -> None:
    setup_logger("step4", level="DEBUG" if args.verbose else "INFO")
    logger.info("=== Step 4 시작: {} (version={}) ===", args.voice_dir, args.version)
    t0 = time.time()
    run_step4(
        args.voice_dir, args.version, args.output_text,
        args.ref_audio, args.ref_text, args.ref_lang,
    )
    voice_name = os.path.basename(os.path.abspath(args.voice_dir))
    logger.info("=== Step 4 완료 ({:.0f}초) ===", time.time() - t0)
    logger.info(
        "[완료] voice.yaml이 생성되었습니다.\n"
        "  합성 테스트: curl -X POST http://localhost:9880/tts -H 'Content-Type: application/json' "
        "-d '{{\"voice\": \"{}\", \"text\": \"테스트\", \"text_lang\": \"ko\"}}' --output test.wav",
        voice_name,
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    setup_logger("pipeline", level="DEBUG" if args.verbose else "INFO")

    voice_dir = args.voice_dir
    version = args.version
    skip = set(args.skip or [])

    total_start = time.time()
    logger.info("=== 파이프라인 시작: {} (version={}) ===", voice_dir, version)

    if "step1" not in skip:
        run_step1(voice_dir, args.config)
    if "step2" not in skip:
        run_step2(voice_dir, version)
    if "step3" not in skip:
        run_step3(voice_dir, version, args.epochs, args.batch_size)
    if "step4" not in skip:
        run_step4(
            voice_dir, version, args.output_text,
            args.ref_audio, args.ref_text, args.ref_lang,
        )

    total_elapsed = time.time() - total_start
    logger.info("=== 파이프라인 완료 ({:.0f}초) ===", total_elapsed)
