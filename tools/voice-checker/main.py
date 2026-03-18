"""Voice Checker: 오디오 품질 이진 분류 도구.

서브커맨드:
  import   — 오디오 파일을 data/에 복사하고 labels.json에 등록
  serve    — 라벨링 UI 서버 실행
  train    — labels.json 기반 CNN 모델 학습
  predict  — 학습된 모델로 오디오 품질 예측
"""
from __future__ import annotations

import argparse
import json
import logging as _logging
import os
import shutil
import sys
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

_PROJECT_ROOT = Path(__file__).resolve().parent
_GPT_SOVITS_ROOT = _PROJECT_ROOT.parents[1]
_DATA_DIR = _GPT_SOVITS_ROOT / "data" / "voice-checker"
_LABELS_FILE = _DATA_DIR / "labels.json"
_MODELS_DIR = _DATA_DIR / "models"
_LOG_DIR = _GPT_SOVITS_ROOT / "logs"
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


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
# labels.json 유틸
# ---------------------------------------------------------------------------


def _load_labels() -> list[dict]:
    """labels.json을 로드한다. 파일이 없으면 빈 리스트를 반환한다."""
    if not _LABELS_FILE.exists():
        return []
    with open(_LABELS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("files", [])


def _save_labels(files: list[dict]) -> None:
    """labels.json을 저장한다."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# import 커맨드
# ---------------------------------------------------------------------------


def _cmd_import(args: argparse.Namespace) -> None:
    """오디오 파일을 data/에 복사하고 labels.json에 등록한다."""
    _setup_logger("voice_checker", level="INFO")

    inp = Path(args.input)
    if inp.is_file():
        audio_files = [inp]
    elif inp.is_dir():
        audio_files = sorted(
            p for p in inp.iterdir()
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
    else:
        logger.error("입력 경로가 유효하지 않습니다: {}", inp)
        sys.exit(1)

    if not audio_files:
        logger.warning("오디오 파일이 없습니다: {}", inp)
        return

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels = _load_labels()
    existing_names = {entry["name"] for entry in labels}

    copied = 0
    skipped = 0
    for src in audio_files:
        name = src.name
        if name in existing_names:
            skipped += 1
            continue
        dst = _DATA_DIR / name
        shutil.copy2(src, dst)
        labels.append({"name": name, "label": "unlabeled"})
        existing_names.add(name)
        copied += 1

    _save_labels(labels)

    counts = {"good": 0, "bad": 0, "unlabeled": 0}
    for entry in labels:
        lbl = entry.get("label", "unlabeled")
        counts[lbl] = counts.get(lbl, 0) + 1

    logger.info(
        "import 완료: 복사 {} / 스킵 {} / 전체 {} (good {} / bad {} / unlabeled {})",
        copied, skipped, len(labels),
        counts["good"], counts["bad"], counts["unlabeled"],
    )


# ---------------------------------------------------------------------------
# serve 커맨드
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> None:
    """라벨링 UI는 메인 서버에 통합되었습니다."""
    _setup_logger("voice_checker", level="INFO")
    logger.warning(
        "serve 커맨드는 메인 서버에 통합되었습니다.\n"
        "  → 프로젝트 루트에서 'python main.py serve' 실행 후\n"
        "  → http://localhost:9880/voice-checker/labeling 접속",
    )


# ---------------------------------------------------------------------------
# train 커맨드
# ---------------------------------------------------------------------------


def _load_vc_config(config_path: Path):
    """루트 conf.yaml의 voice_checker 섹션에서 Config를 로드한다."""
    from src.config.config import Config, load_config

    if not config_path.exists():
        return Config()

    # 루트 conf.yaml에서 voice_checker 섹션 추출
    import yaml
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    vc = raw.get("voice_checker", {})
    return load_config_from_dict(vc)


def load_config_from_dict(raw: dict):
    """dict에서 voice-checker Config를 생성한다."""
    from src.config.config import (
        AudioConfig, AugmentationConfig, Config,
        InferenceConfig, ServiceConfig, TrainingConfig,
    )
    return Config(
        audio=AudioConfig(**raw["audio"]) if "audio" in raw else AudioConfig(),
        training=TrainingConfig(**raw["training"]) if "training" in raw else TrainingConfig(),
        augmentation=AugmentationConfig(**raw["augmentation"]) if "augmentation" in raw else AugmentationConfig(),
        inference=InferenceConfig(**raw["inference"]) if "inference" in raw else InferenceConfig(),
        service=ServiceConfig(**raw["service"]) if "service" in raw else ServiceConfig(),
        log_level=raw.get("log_level", "INFO"),
    )


def _cmd_train(args: argparse.Namespace) -> None:
    """labels.json 기반으로 CNN 모델을 학습한다."""
    from src.training.trainer import run_training

    config = _load_vc_config(Path(args.config))

    _setup_logger(
        "voice_checker",
        level="DEBUG" if args.verbose else config.log_level,
        log_dir=_LOG_DIR,
    )

    run_training(config, _DATA_DIR, _LABELS_FILE, _MODELS_DIR)


# ---------------------------------------------------------------------------
# predict 커맨드
# ---------------------------------------------------------------------------


def _cmd_predict(args: argparse.Namespace) -> None:
    """학습된 모델로 오디오 품질을 예측한다."""
    from src.inference.predictor import VoiceQualityPredictor

    config = _load_vc_config(Path(args.config))

    _setup_logger(
        "voice_checker",
        level="DEBUG" if args.verbose else config.log_level,
    )

    model_path = args.model or str(_MODELS_DIR / config.inference.model_path)
    if not os.path.exists(model_path):
        logger.error("모델 파일을 찾을 수 없습니다: {}", model_path)
        sys.exit(1)

    predictor = VoiceQualityPredictor(model_path, config)
    threshold = args.threshold or config.inference.threshold

    inp = Path(args.input)
    if inp.is_file():
        targets = [inp]
    elif inp.is_dir():
        targets = sorted(
            p for p in inp.iterdir()
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
    else:
        logger.error("입력 경로가 유효하지 않습니다: {}", inp)
        sys.exit(1)

    good_count = 0
    bad_count = 0
    for path in targets:
        is_good, confidence = predictor.predict(str(path), threshold)
        label = "good" if is_good else "bad"
        if is_good:
            good_count += 1
        else:
            bad_count += 1
        logger.info("{}: {} (confidence={:.3f})", path.name, label, confidence)

    if len(targets) > 1:
        logger.info("결과: good {} / bad {} / 전체 {}", good_count, bad_count, len(targets))


# ---------------------------------------------------------------------------
# 메인 파서
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Voice Checker — 오디오 품질 이진 분류 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- import ---
    sp_import = sub.add_parser("import", help="오디오 파일을 data/에 수집")
    sp_import.add_argument("input", help="오디오 파일 또는 폴더 경로")

    # --- serve ---
    sp_serve = sub.add_parser("serve", help="라벨링 UI 서버 실행")
    sp_serve.add_argument("-v", "--verbose", action="store_true")
    sp_serve.add_argument("-c", "--config", default="conf.yaml")
    sp_serve.add_argument("--host", default=None)
    sp_serve.add_argument("--port", type=int, default=None)

    # --- train ---
    sp_train = sub.add_parser("train", help="CNN 모델 학습")
    sp_train.add_argument("-v", "--verbose", action="store_true")
    sp_train.add_argument("-c", "--config", default="conf.yaml")

    # --- predict ---
    sp_predict = sub.add_parser("predict", help="오디오 품질 예측")
    sp_predict.add_argument("input", help="오디오 파일 또는 폴더 경로")
    sp_predict.add_argument("-v", "--verbose", action="store_true")
    sp_predict.add_argument("-c", "--config", default="conf.yaml")
    sp_predict.add_argument("-m", "--model", default=None, help="모델 파일 경로")
    sp_predict.add_argument("-t", "--threshold", type=float, default=None, help="분류 임계값")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "import": _cmd_import,
        "serve": _cmd_serve,
        "train": _cmd_train,
        "predict": _cmd_predict,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
