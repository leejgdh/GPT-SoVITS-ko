"""voice 작업 폴더의 생성 · 단계 파일 · 음성 인식 목록 관리."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import BinaryIO

import yaml
from loguru import logger

from src.exceptions import (
    InvalidFileNameError,
    InvalidLabelListError,
    StageFileExistsError,
    UnsupportedFileError,
    VoiceExistsError,
)

RAW_AUDIO_DIR = "raw_audio"
VOCAL_AUDIO_DIR = "step1/03_vocal"
_VOCAL_LABEL_FILE = "step1/04_asr/03_vocal.list"
_LABEL_CATEGORY = "03_vocal"
_LABEL_STATES = frozenset({"pending", "approved", "rejected"})
_LABEL_BREAKERS = ("|", "\t", "\n", "\r")
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


def remove_voice(voices_dir: str, name: str) -> None:
    """voice 폴더를 학습 자료 · 산출물째 지운다."""
    voice_dir = Path(voices_dir) / name
    shutil.rmtree(voice_dir)
    logger.info("voice 삭제: {}", voice_dir)


def save_stage_file(directory: Path, filename: str, stream: BinaryIO, accepts: frozenset[str]) -> None:
    """파일을 단계 폴더에 저장한다. 폴더가 없으면 만든다.

    ``accepts`` 에 없는 확장자면 ``UnsupportedFileError``, 목록 파일을 깨뜨리는 문자가 이름에 있으면
    ``InvalidFileNameError``, 같은 이름이 있으면 ``StageFileExistsError``.
    """
    if Path(filename).suffix.lower() not in accepts:
        raise UnsupportedFileError(f"이 단계에 올릴 수 없는 파일입니다: {filename}")
    if any(ch in filename for ch in _LABEL_BREAKERS):
        raise InvalidFileNameError(f"파일 이름에 쓸 수 없는 문자가 있습니다: {filename}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        raise StageFileExistsError(f"같은 이름의 파일이 이미 있습니다: {filename}")
    with open(target, "wb") as f:
        shutil.copyfileobj(stream, f)
    logger.info("파일 저장: {}", target)


def remove_stage_file(directory: Path, filename: str) -> bool:
    """단계 폴더의 파일을 지운다. 지웠으면 True, 없으면 False."""
    target = directory / filename
    if not target.is_file():
        return False
    target.unlink()
    logger.info("파일 삭제: {}", target)
    return True


def save_label_list(voice_dir: Path, voice_dir_ref: str, stream: BinaryIO) -> int:
    """올린 음성 인식 목록을 ``step1/04_asr`` 의 목록 파일로 저장한다. 저장한 줄 수를 돌려준다.

    각 줄은 ``파일|구분|언어|문장[|상태]`` 이다. 파일 칸은 이름만 쓰며, 잘라낸 음성 단계에 그 파일이 있어야 한다.
    저장할 때 파일 칸은 ``voice_dir_ref``(작업 폴더 기준 voice 경로) 아래 잘라낸 음성의 경로로 바꾸고,
    상태가 없으면 ``pending`` 으로 둔다. 형식이 맞지 않거나 없는 음성을 가리키면 ``InvalidLabelListError``,
    목록 파일이 이미 있으면 ``StageFileExistsError``.
    """
    target = voice_dir / _VOCAL_LABEL_FILE
    if target.exists():
        raise StageFileExistsError("음성 인식 결과가 이미 있습니다. 지운 뒤에 올려 주세요")
    vocal_dir = voice_dir / VOCAL_AUDIO_DIR
    lines: list[str] = []
    missing: list[str] = []
    for number, raw in enumerate(stream.read().decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("|")
        if len(parts) < 4:
            raise InvalidLabelListError(f"{number}번째 줄의 형식이 맞지 않습니다")
        name = re.split(r"[\\/]", parts[0].strip())[-1]
        state = parts[4].strip() if len(parts) >= 5 and parts[4].strip() else "pending"
        if state not in _LABEL_STATES:
            raise InvalidLabelListError(f"{number}번째 줄의 상태가 올바르지 않습니다: {state}")
        if not (vocal_dir / name).is_file():
            missing.append(name)
            continue
        lines.append(f"{voice_dir_ref}/{VOCAL_AUDIO_DIR}/{name}|{_LABEL_CATEGORY}|{parts[2].strip()}| {parts[3].strip()}|{state}")
    if missing:
        raise InvalidLabelListError(f"목록이 가리키는 음성 {len(missing)}개가 잘라낸 음성에 없습니다: {', '.join(missing[:3])}")
    if not lines:
        raise InvalidLabelListError("목록에 올릴 줄이 없습니다")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    logger.info("음성 인식 목록 저장: {} ({}줄)", target, len(lines))
    return len(lines)


def remove_vocal_label(voice_dir: Path, filename: str) -> int:
    """음성 인식 목록에서 ``filename`` 을 가리키는 줄을 지운다. 지운 줄 수를 돌려준다."""
    label_path = voice_dir / _VOCAL_LABEL_FILE
    if not label_path.is_file():
        return 0
    lines = label_path.read_text(encoding="utf-8").split("\n")
    kept = [line for line in lines if Path(line.split("|", 1)[0]).name != filename]
    removed = len(lines) - len(kept)
    if removed:
        label_path.write_text("\n".join(kept), encoding="utf-8")
        logger.info("인식 목록에서 삭제: {} ({}줄)", filename, removed)
    return removed

