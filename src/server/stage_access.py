"""요청에서 voice 폴더와 학습 단계 폴더를 찾는다 — 경로 검증을 한 곳에 모은다."""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

from src.dataset.stages import STAGES, Stage
from src.server.context import ServiceContext
from src.server.paths import validate_name


def voice_dir_of(request: Request, name: str) -> Path:
    """voice 폴더. 이름이 올바르지 않거나 폴더가 없으면 HTTP 오류."""
    ctx: ServiceContext = request.app.state.context
    voice_dir = Path(ctx.config.voices_dir) / validate_name(name, "voice 이름")
    if not voice_dir.is_dir():
        raise HTTPException(404, detail=f"voice '{name}' 폴더가 없습니다")
    return voice_dir


def stage_of(key: str) -> Stage:
    """단계 키에 맞는 단계. 모르는 키면 HTTP 오류."""
    stage = STAGES.get(key)
    if stage is None:
        raise HTTPException(404, detail=f"알 수 없는 단계: {key}")
    return stage


def stage_dir_of(voice_dir: Path, stage: Stage, version: str | None, *, create: bool = False) -> Path:
    """단계 폴더. ``create`` 이면 없을 때 만들고, 아니면 없을 때 HTTP 오류."""
    if stage.versioned:
        if not version:
            raise HTTPException(400, detail=f"단계 '{stage.key}' 는 version 이 필요합니다")
        validate_name(version, "version")
    directory = stage.directory(voice_dir, version)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise HTTPException(404, detail=f"단계 '{stage.key}' 폴더가 없습니다")
    return directory
