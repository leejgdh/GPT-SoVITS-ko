"""voice 작업 폴더의 생성과 원본 오디오 관리."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import BinaryIO

import yaml
from loguru import logger

from src.dataset.files import AUDIO_SUFFIXES
from src.exceptions import RawAudioExistsError, UnsupportedAudioError, VoiceExistsError

RAW_AUDIO_DIR = "raw_audio"
_VOICE_YAML = "voice.yaml"


def ensure_voice_yaml(voice_dir: str) -> None:
    """voice.yaml 이 없으면 ``available: false`` 로 초기 파일을 만든다."""
    yaml_path = os.path.join(voice_dir, _VOICE_YAML)
    if os.path.exists(yaml_path):
        return
    name = os.path.basename(os.path.abspath(voice_dir))
    os.makedirs(voice_dir, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"name": name, "available": False}, f, allow_unicode=True, sort_keys=False)
    logger.info("voice.yaml 초기 생성: {} (available: false)", yaml_path)


def create_voice(voices_dir: str, name: str) -> Path:
    """voice 폴더와 초기 voice.yaml 을 만든다. 이미 있으면 ``VoiceExistsError``."""
    voice_dir = Path(voices_dir) / name
    if voice_dir.exists():
        raise VoiceExistsError(f"voice '{name}' 이(가) 이미 있습니다")
    (voice_dir / RAW_AUDIO_DIR).mkdir(parents=True)
    ensure_voice_yaml(str(voice_dir))
    return voice_dir


def save_raw_audio(voice_dir: Path, filename: str, stream: BinaryIO) -> None:
    """원본 오디오를 ``raw_audio`` 에 저장한다.

    오디오 확장자가 아니면 ``UnsupportedAudioError``, 같은 이름이 있으면 ``RawAudioExistsError``.
    """
    if Path(filename).suffix.lower() not in AUDIO_SUFFIXES:
        raise UnsupportedAudioError(f"오디오 파일이 아닙니다: {filename}")
    target_dir = voice_dir / RAW_AUDIO_DIR
    target_dir.mkdir(exist_ok=True)
    target = target_dir / filename
    if target.exists():
        raise RawAudioExistsError(f"같은 이름의 파일이 이미 있습니다: {filename}")
    with open(target, "wb") as f:
        shutil.copyfileobj(stream, f)
    logger.info("원본 오디오 저장: {}", target)


def remove_raw_audio(voice_dir: Path, filename: str) -> bool:
    """원본 오디오를 지운다. 지웠으면 True, 없으면 False."""
    target = voice_dir / RAW_AUDIO_DIR / filename
    if not target.is_file():
        return False
    target.unlink()
    logger.info("원본 오디오 삭제: {}", target)
    return True
