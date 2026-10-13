"""Voice Checker 분류 모델 현황 API."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.server.context import ServiceContext

router = APIRouter(prefix="/voice-checker", tags=["voice-checker"])


class VoiceCheckerModel(BaseModel):
    enabled: bool
    exists: bool
    size: int | None
    modified: float | None
    threshold: float | None


@router.get("/model", response_model=VoiceCheckerModel)
def get_model(request: Request) -> VoiceCheckerModel:
    """분류 모델 파일의 존재 · 크기 · 수정 시각(UTC epoch 초)과 분류 임계값. 설정에 없으면 ``enabled`` 가 false 다."""
    ctx: ServiceContext = request.app.state.context
    vc = ctx.config.voice_checker
    if vc is None:
        return VoiceCheckerModel(enabled=False, exists=False, size=None, modified=None, threshold=None)
    path = Path(vc.inference.model_path)
    stat = path.stat() if path.is_file() else None
    return VoiceCheckerModel(
        enabled=True,
        exists=stat is not None,
        size=stat.st_size if stat else None,
        modified=stat.st_mtime if stat else None,
        threshold=vc.inference.threshold,
    )
