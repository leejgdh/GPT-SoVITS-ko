"""voice 작업 폴더 관리 API — 생성 · 재스캔 · 단계별 파일 올리기/지우기 · 학습 산출물 정리."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, field_validator

from src.exceptions import (
    CleanupError,
    InvalidFileNameError,
    InvalidLabelListError,
    StageFileExistsError,
    UnsupportedFileError,
    VoiceExistsError,
)
from src.server.context import ServiceContext
from src.server.paths import validate_name
from src.server.stage_access import stage_dir_of, stage_of, voice_dir_of
from src.voices.cleanup import CleanupPlan, execute_cleanup, plan_cleanup
from src.voices.workspace import create_voice, remove_stage_file, remove_vocal_label, save_label_list, save_stage_file

router = APIRouter(tags=["voice-manage"])

_VOICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


# ---------------------------------------------------------------------------
# 요청 · 응답 모델
# ---------------------------------------------------------------------------

class CreateVoiceRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _VOICE_NAME.match(v):
            msg = "이름은 영문자·숫자·'-'·'_' 로 64자 이내여야 합니다"
            raise ValueError(msg)
        return v


class VoiceCreated(BaseModel):
    name: str
    available: bool


class VoicesReloaded(BaseModel):
    voices: int


class FilesUploaded(BaseModel):
    saved: list[str]


class CleanupRequest(BaseModel):
    dry_run: bool
    keep_raw: bool
    keep_asr: bool


class CleanupTargetInfo(BaseModel):
    path: str
    bytes: int


class CleanupResult(BaseModel):
    executed: bool
    preserve: list[str]
    targets: list[CleanupTargetInfo]
    total_bytes: int


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _context(request: Request) -> ServiceContext:
    return request.app.state.context


def _require_idle(request: Request, name: str) -> None:
    if request.app.state.jobs.is_voice_busy(name):
        raise HTTPException(409, detail=f"voice '{name}' 에 대기 중이거나 실행 중인 job 이 있습니다")


def _result(voice_dir: Path, plan: CleanupPlan, executed: bool) -> CleanupResult:
    def rel(path: str) -> str:
        return os.path.relpath(path, voice_dir)

    return CleanupResult(
        executed=executed,
        preserve=[rel(p) for p in plan.preserve],
        targets=[CleanupTargetInfo(path=rel(t.path), bytes=t.bytes) for t in plan.targets],
        total_bytes=plan.total_bytes,
    )


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.post("/voices", status_code=201, response_model=VoiceCreated)
async def create_voice_endpoint(request: Request, body: CreateVoiceRequest) -> VoiceCreated:
    """새 voice 폴더를 만든다. 학습이 끝나기 전까지 ``available`` 은 false 다."""
    ctx = _context(request)
    try:
        await asyncio.to_thread(create_voice, ctx.config.voices_dir, body.name)
    except VoiceExistsError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    await ctx.reload_voices()
    return VoiceCreated(name=body.name, available=False)


@router.post("/voices/reload", response_model=VoicesReloaded)
async def reload_voices(request: Request) -> VoicesReloaded:
    """voice 폴더를 다시 읽는다. 학습이 끝난 voice 를 합성에 반영할 때 쓴다."""
    return VoicesReloaded(voices=await _context(request).reload_voices())


@router.post("/voices/{name}/dataset/{stage}/files", response_model=FilesUploaded)
async def upload_stage_files(
    request: Request, name: str, stage: str, version: str | None = None, files: list[UploadFile] = File(...),
) -> FilesUploaded:
    """학습 단계 폴더에 파일을 올린다. 단계마다 받는 확장자가 다르다 (``GET /voices/{name}/dataset`` 의 ``accepts``).

    ``asr`` 단계는 음성 인식 목록(``파일|구분|언어|문장[|상태]``)을 받아 잘라낸 음성의 경로로 맞춰 저장한다.
    """
    voice_dir = voice_dir_of(request, name)
    spec = stage_of(stage)
    if not spec.accepts:
        raise HTTPException(400, detail=f"단계 '{stage}' 에는 파일을 올릴 수 없습니다")
    _require_idle(request, name)
    directory = stage_dir_of(voice_dir, spec, version, create=True)
    ctx: ServiceContext = request.app.state.context
    saved: list[str] = []
    for upload in files:
        filename = validate_name(os.path.basename(upload.filename or ""), "파일명")
        try:
            if stage == "asr":
                await asyncio.to_thread(save_label_list, voice_dir, os.path.join(ctx.config.voices_dir, name), upload.file)
                filename = "03_vocal.list"
            else:
                await asyncio.to_thread(save_stage_file, directory, filename, upload.file, spec.accepts)
        except UnsupportedFileError as exc:
            raise HTTPException(415, detail=str(exc)) from exc
        except (InvalidFileNameError, InvalidLabelListError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        except StageFileExistsError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        saved.append(filename)
    return FilesUploaded(saved=saved)


@router.delete("/voices/{name}/dataset/{stage}/files/{filename}", status_code=204)
async def delete_stage_file(request: Request, name: str, stage: str, filename: str, version: str | None = None) -> None:
    """학습 단계 폴더의 파일을 지운다. 잘라낸 음성이면 음성 인식 목록에서 그 파일의 줄도 지운다."""
    voice_dir = voice_dir_of(request, name)
    spec = stage_of(stage)
    if not spec.accepts:
        raise HTTPException(400, detail=f"단계 '{stage}' 의 파일은 지울 수 없습니다")
    _require_idle(request, name)
    directory = stage_dir_of(voice_dir, spec, version)
    if not await asyncio.to_thread(remove_stage_file, directory, validate_name(filename, "파일명")):
        raise HTTPException(404, detail=f"파일 없음: {filename}")
    if stage == "vocal":
        await asyncio.to_thread(remove_vocal_label, voice_dir, filename)


@router.post("/voices/{name}/cleanup", response_model=CleanupResult)
async def cleanup_voice(request: Request, name: str, body: CleanupRequest) -> CleanupResult:
    """학습 산출물을 정리한다. ``dry_run`` 이면 삭제 대상만 돌려주고 지우지 않는다."""
    voice_dir = voice_dir_of(request, name)
    try:
        plan = await asyncio.to_thread(
            plan_cleanup, str(voice_dir), keep_raw=body.keep_raw, keep_asr=body.keep_asr,
        )
    except CleanupError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    if not body.dry_run:
        _require_idle(request, name)
        await asyncio.to_thread(execute_cleanup, str(voice_dir), plan)
    return _result(voice_dir, plan, executed=not body.dry_run)
