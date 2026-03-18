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


# ---------------------------------------------------------------------------
# 개별 실행 함수
# ---------------------------------------------------------------------------


def run_denoise(voice_dir: str) -> None:
    _run([sys.executable, "scripts/data_preparation/denoise.py",
          "--voice-dir", voice_dir], "denoise")


def run_slice(voice_dir: str) -> None:
    _run([sys.executable, "scripts/data_preparation/slice_audio.py",
          "--voice-dir", voice_dir], "slice")


def run_uvr5(voice_dir: str) -> None:
    _run([sys.executable, "scripts/data_preparation/uvr5_separate.py",
          "--voice-dir", voice_dir], "uvr5")


def run_asr(voice_dir: str) -> None:
    _run([sys.executable, "scripts/data_preparation/asr_whisper.py",
          "--voice-dir", voice_dir], "asr")


def run_classify(voice_dir: str, config_path: str = "conf.yaml") -> None:
    """vocal.list의 pending 상태를 Voice Checker CNN으로 재분류한다."""
    from src.config.config import VoiceCheckerConfig, load_config

    path = Path(config_path)
    config = load_config(path) if path.exists() else None
    if config is None or config.voice_checker is None:
        logger.info("voice_checker 미설정 — classify 건너뜀")
        return

    model_path = config.voice_checker.inference.model_path
    if not Path(model_path).exists():
        logger.warning("Voice Checker 모델 없음: {} — classify 건너뜀", model_path)
        return

    # voice-checker predictor 로드
    vc_root = os.path.join(_PROJECT_ROOT, "tools", "voice-checker")
    if vc_root not in sys.path:
        sys.path.insert(0, vc_root)
    from vc.predictor import VoiceQualityPredictor

    predictor = VoiceQualityPredictor(model_path, config.voice_checker)

    # vocal.list 읽기
    asr_dir = os.path.join(voice_dir, "step1", "04_asr")
    list_files = glob.glob(os.path.join(asr_dir, "*.list"))
    if not list_files:
        logger.error("ASR 라벨 파일이 없습니다: {}", asr_dir)
        sys.exit(1)

    for list_file in list_files:
        with open(list_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated = 0
        output = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) < 5:
                output.append(line.rstrip())
                continue

            audio_path, category, lang, text, state = parts[0], parts[1], parts[2], parts[3], parts[4]

            if state != "pending":
                output.append(line.rstrip())
                continue

            is_good, confidence = predictor.predict(audio_path)
            if is_good:
                new_state = "approved"
            else:
                new_state = "rejected"
            output.append(f"{audio_path}|{category}|{lang}|{text}|{new_state}")
            updated += 1
            logger.debug("{} -> {} (confidence={:.3f})", os.path.basename(audio_path), new_state, confidence)

        with open(list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output) + "\n")

        logger.info("{}: {} 건 재분류 완료", os.path.basename(list_file), updated)


def run_get_text(voice_dir: str, version: str) -> None:
    _run([sys.executable, "scripts/preprocessing/get_text.py",
          "--voice-dir", voice_dir, "--version", version], "get-text")


def run_get_hubert(voice_dir: str, version: str) -> None:
    _run([sys.executable, "scripts/preprocessing/get_hubert_wav32k.py",
          "--voice-dir", voice_dir, "--version", version], "get-hubert")


def run_get_sv(voice_dir: str, version: str) -> None:
    _run([sys.executable, "scripts/preprocessing/get_sv.py",
          "--voice-dir", voice_dir, "--version", version], "get-sv")


def run_get_semantic(voice_dir: str, version: str) -> None:
    _run([sys.executable, "scripts/preprocessing/get_semantic.py",
          "--voice-dir", voice_dir, "--version", version], "get-semantic")


def run_train_gpt(
    voice_dir: str, version: str,
    epochs: int | None = None, batch_size: int | None = None,
) -> None:
    cmd = [sys.executable, "scripts/training/s1_train.py",
           "--voice-dir", voice_dir, "--version", version]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    _run(cmd, "GPT train")


def run_train_sovits(
    voice_dir: str, version: str,
    epochs: int | None = None, batch_size: int | None = None,
) -> None:
    sovits_script = _SOVITS_TRAIN_SCRIPT[version]
    cmd = [sys.executable, sovits_script,
           "--voice-dir", voice_dir, "--version", version]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    _run(cmd, "SoVITS train")


# ---------------------------------------------------------------------------
# step 묶음 실행 함수
# ---------------------------------------------------------------------------


def run_step1(voice_dir: str, config_path: str = "conf.yaml") -> None:
    _ensure_voice_yaml(voice_dir)
    run_denoise(voice_dir)
    run_slice(voice_dir)
    run_uvr5(voice_dir)
    run_asr(voice_dir)
    run_classify(voice_dir, config_path)


def run_step2(voice_dir: str, version: str) -> None:
    _clean_step_dir(voice_dir, "step2", version, "전처리")
    run_get_text(voice_dir, version)
    run_get_hubert(voice_dir, version)
    if version in _SV_VERSIONS:
        run_get_sv(voice_dir, version)
    run_get_semantic(voice_dir, version)


def run_step3(
    voice_dir: str, version: str,
    epochs: int | None = None, batch_size: int | None = None,
) -> None:
    _clean_step_dir(voice_dir, "step3", version, "학습")
    run_train_gpt(voice_dir, version, epochs, batch_size)
    run_train_sovits(voice_dir, version, epochs, batch_size)


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


def _log_step(name: str, args: argparse.Namespace):
    setup_logger(name, level="DEBUG" if args.verbose else "INFO")


# ── 하위 커맨드 ──


def cmd_denoise(args: argparse.Namespace) -> None:
    _log_step("denoise", args)
    run_denoise(args.voice_dir)


def cmd_slice(args: argparse.Namespace) -> None:
    _log_step("slice", args)
    run_slice(args.voice_dir)


def cmd_uvr5(args: argparse.Namespace) -> None:
    _log_step("uvr5", args)
    run_uvr5(args.voice_dir)


def cmd_asr(args: argparse.Namespace) -> None:
    _log_step("asr", args)
    run_asr(args.voice_dir)


def cmd_classify(args: argparse.Namespace) -> None:
    _log_step("classify", args)
    run_classify(args.voice_dir, args.config)


def cmd_get_text(args: argparse.Namespace) -> None:
    _log_step("get-text", args)
    run_get_text(args.voice_dir, args.version)


def cmd_get_hubert(args: argparse.Namespace) -> None:
    _log_step("get-hubert", args)
    run_get_hubert(args.voice_dir, args.version)


def cmd_get_sv(args: argparse.Namespace) -> None:
    _log_step("get-sv", args)
    run_get_sv(args.voice_dir, args.version)


def cmd_get_semantic(args: argparse.Namespace) -> None:
    _log_step("get-semantic", args)
    run_get_semantic(args.voice_dir, args.version)


def cmd_train_gpt(args: argparse.Namespace) -> None:
    _log_step("train-gpt", args)
    run_train_gpt(args.voice_dir, args.version, args.epochs, args.batch_size)


def cmd_train_sovits(args: argparse.Namespace) -> None:
    _log_step("train-sovits", args)
    run_train_sovits(args.voice_dir, args.version, args.epochs, args.batch_size)


# ── step 묶음 커맨드 ──


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
