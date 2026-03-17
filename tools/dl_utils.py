# -*- coding: utf-8 -*-
"""모델 다운로드 유틸리티.

HuggingFace Hub / ModelScope에서 모델 가중치를 다운로드한다.
로컬에 이미 존재하면 스킵한다.
"""
from __future__ import annotations

import os
import shutil

import requests
from loguru import logger


def is_huggingface_accessible(timeout: float = 3.0) -> bool:
    """HuggingFace API 접근 가능 여부를 확인한다."""
    try:
        requests.get("https://huggingface.co/api/models/gpt2", timeout=timeout)
        return True
    except Exception:
        return False


def download_from_hf(
    repo_id: str,
    *,
    filename: str | None = None,
    subfolder: str | None = None,
    local_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """HuggingFace Hub에서 모델을 다운로드한다.

    Args:
        repo_id: HuggingFace 레포 ID (예: "lj1995/VoiceConversionWebUI")
        filename: 단일 파일 다운로드 시 파일명 (지정 시 hf_hub_download)
        subfolder: 레포 내 하위 폴더 경로
        local_dir: snapshot_download 시 로컬 저장 경로
        allow_patterns: snapshot_download 시 필터 패턴 목록

    Returns:
        다운로드된 파일/디렉토리 경로 (캐시 경로 또는 local_dir)
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename:
        path_desc = f"{subfolder}/{filename}" if subfolder else filename
        logger.info("HuggingFace 다운로드: {}/{}", repo_id, path_desc)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        )
    else:
        logger.info("HuggingFace 다운로드: {} -> {}", repo_id, local_dir)
        return snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )


def download_from_modelscope(
    model_id: str,
    local_dir: str,
    **kwargs,
) -> str:
    """ModelScope에서 모델을 다운로드한다.

    Args:
        model_id: ModelScope 모델 ID
        local_dir: 로컬 저장 경로

    Returns:
        다운로드된 디렉토리 경로
    """
    from modelscope import snapshot_download

    logger.info("ModelScope 다운로드: {} -> {}", model_id, local_dir)
    return snapshot_download(model_id, local_dir=local_dir, **kwargs)


def ensure_file(
    local_path: str,
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    source: str = "hf",
) -> str:
    """로컬에 파일이 없으면 다운로드한다.

    HF 캐시에 다운로드 후, 원하는 로컬 경로로 복사한다.
    HF 파일명과 로컬 파일명이 달라도 동작한다.

    Args:
        local_path: 기대하는 로컬 파일 경로
        repo_id: 레포 ID
        filename: 레포 내 파일명
        subfolder: 레포 내 하위 폴더
        source: "hf" 또는 "modelscope"

    Returns:
        로컬 파일 경로
    """
    if os.path.exists(local_path):
        return local_path

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    if source == "hf":
        cached_path = download_from_hf(repo_id, filename=filename, subfolder=subfolder)
        shutil.copy2(cached_path, local_path)
    else:
        download_from_modelscope(repo_id, local_dir)

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"다운로드 후에도 파일을 찾을 수 없습니다: {local_path}")

    return local_path


def ensure_dir(
    local_dir: str,
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    source: str = "hf",
) -> str:
    """로컬에 디렉토리(또는 내용)가 없으면 다운로드한다.

    Args:
        local_dir: 기대하는 로컬 디렉토리 경로
        repo_id: 레포 ID
        allow_patterns: 다운로드할 파일 패턴 목록
        source: "hf" 또는 "modelscope"

    Returns:
        로컬 디렉토리 경로
    """
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir

    if source == "hf":
        download_from_hf(repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
    else:
        download_from_modelscope(repo_id, local_dir)

    return local_dir


# ---------------------------------------------------------------------------
# Whisper ASR 모델 다운로드
# ---------------------------------------------------------------------------

_WHISPER_FILE_BASE = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
_WHISPER_FILE_V3 = ["config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"]


def download_whisper_model(model_size: str) -> str:
    """Faster Whisper 모델을 다운로드하고 경로를 반환한다.

    Args:
        model_size: 모델 크기 (예: "large-v3", "medium", "large-v3-turbo")

    Returns:
        모델 디렉토리 경로
    """
    source = "HF" if is_huggingface_accessible() else "ModelScope"

    if source == "HF":
        repo_id, model_path = _resolve_whisper_hf(model_size)
        files = _WHISPER_FILE_V3 if ("large-v3" in model_size or "distil" in model_size) else _WHISPER_FILE_BASE
        download_from_hf(repo_id, local_dir=model_path, allow_patterns=files)
        return model_path
    else:
        repo_id = "XXXXRT/faster-whisper"
        model_path = "data/models/asr"
        files = _WHISPER_FILE_V3 if ("large-v3" in model_size or "distil" in model_size) else _WHISPER_FILE_BASE
        ms_files = [f"faster-whisper-{model_size}/{f}".replace("whisper-distil", "distil-whisper") for f in files]
        download_from_modelscope(repo_id, local_dir=model_path, allow_patterns=ms_files)
        return os.path.join(model_path, f"faster-whisper-{model_size}".replace("whisper-distil", "distil-whisper"))


def _resolve_whisper_hf(model_size: str) -> tuple[str, str]:
    """HuggingFace용 Whisper repo_id와 로컬 경로를 결정한다."""
    if "distil" in model_size:
        if "3.5" in model_size:
            return "distil-whisper/distil-large-v3.5-ct2", "data/models/asr/faster-distil-whisper-large-v3.5"
        parts = model_size.split("-", maxsplit=1)
        repo_id = f"Systran/faster-{parts[0]}-whisper-{parts[1]}"
        return repo_id, f"data/models/asr/{repo_id.replace('Systran/', '').replace('distil-whisper/', '', 1)}"
    if model_size == "large-v3-turbo":
        return "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "data/models/asr/faster-whisper-large-v3-turbo"
    repo_id = f"Systran/faster-whisper-{model_size}"
    return repo_id, f"data/models/asr/{repo_id.replace('Systran/', '')}"
