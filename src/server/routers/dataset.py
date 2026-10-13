"""voice 별 학습 데이터(단계별 산출물) 열람 API — 읽기 전용."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.dataset.files import AUDIO_SUFFIXES, TEXT_SUFFIXES, list_files, summarize
from src.dataset.stages import STAGES, Stage, voice_versions
from src.dataset.tables import read_curves, read_semantic_rows, read_text_rows
from src.server.context import ServiceContext
from src.server.paths import is_within, validate_name

router = APIRouter(tags=["dataset"])

_MAX_PAGE = 500


# ---------------------------------------------------------------------------
# 응답 모델
# ---------------------------------------------------------------------------

class StageInfo(BaseModel):
    key: str
    version: str | None
    exists: bool
    files: int
    bytes: int


class DatasetSummary(BaseModel):
    voice: str
    versions: list[str]
    stages: list[StageInfo]


class FileInfo(BaseModel):
    name: str
    size: int
    modified: float
    duration: float | None


class FileList(BaseModel):
    total: int
    items: list[FileInfo]


class TextRowInfo(BaseModel):
    name: str
    phonemes: str
    word2ph: str
    text: str


class TextRowList(BaseModel):
    total: int
    items: list[TextRowInfo]


class SemanticRowInfo(BaseModel):
    name: str
    token_count: int
    preview: list[int]


class SemanticRowList(BaseModel):
    total: int
    items: list[SemanticRowInfo]


class CurveInfo(BaseModel):
    source: str
    tag: str
    points: list[tuple[int, float]]


class CurveList(BaseModel):
    available: bool
    curves: list[CurveInfo]


# ---------------------------------------------------------------------------
# 경로 해석
# ---------------------------------------------------------------------------

def _voice_dir(request: Request, name: str) -> Path:
    ctx: ServiceContext = request.app.state.context
    voice_dir = Path(ctx.config.voices_dir) / validate_name(name, "voice 이름")
    if not voice_dir.is_dir():
        raise HTTPException(404, detail=f"voice '{name}' 폴더가 없습니다")
    return voice_dir


def _stage(key: str) -> Stage:
    stage = STAGES.get(key)
    if stage is None:
        raise HTTPException(404, detail=f"알 수 없는 단계: {key}")
    return stage


def _stage_dir(voice_dir: Path, stage: Stage, version: str | None) -> Path:
    if stage.versioned:
        if not version:
            raise HTTPException(400, detail=f"단계 '{stage.key}' 는 version 이 필요합니다")
        validate_name(version, "version")
    directory = stage.directory(voice_dir, version)
    if not directory.is_dir():
        raise HTTPException(404, detail=f"단계 '{stage.key}' 폴더가 없습니다")
    return directory


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.get("/voices/{name}/dataset", response_model=DatasetSummary)
def dataset_summary(request: Request, name: str) -> DatasetSummary:
    """단계별 폴더 존재 여부 · 파일 수 · 용량을 반환한다."""
    voice_dir = _voice_dir(request, name)
    versions = voice_versions(voice_dir)
    infos: list[StageInfo] = []
    for stage in STAGES.values():
        for version in (versions if stage.versioned else [None]):
            result = summarize(stage.directory(voice_dir, version), stage.recursive)
            infos.append(StageInfo(
                key=stage.key, version=version,
                exists=result.exists, files=result.files, bytes=result.bytes,
            ))
    return DatasetSummary(voice=name, versions=versions, stages=infos)


@router.get("/voices/{name}/dataset/text", response_model=TextRowList)
def dataset_text(
    request: Request, name: str, version: str,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=_MAX_PAGE),
) -> TextRowList:
    """step2 음소 파일(name2text.txt)의 한 페이지."""
    directory = _stage_dir(_voice_dir(request, name), STAGES["step2"], version)
    path = directory / "name2text.txt"
    if not path.is_file():
        raise HTTPException(404, detail="음소 파일이 없습니다")
    total, rows = read_text_rows(path, offset, limit)
    return TextRowList(total=total, items=[TextRowInfo(**vars(r)) for r in rows])


@router.get("/voices/{name}/dataset/semantic", response_model=SemanticRowList)
def dataset_semantic(
    request: Request, name: str, version: str,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=_MAX_PAGE),
) -> SemanticRowList:
    """step2 semantic 토큰 파일(name2semantic.tsv)의 한 페이지."""
    directory = _stage_dir(_voice_dir(request, name), STAGES["step2"], version)
    path = directory / "name2semantic.tsv"
    if not path.is_file():
        raise HTTPException(404, detail="semantic 파일이 없습니다")
    total, rows = read_semantic_rows(path, offset, limit)
    return SemanticRowList(total=total, items=[SemanticRowInfo(**vars(r)) for r in rows])


@router.get("/voices/{name}/dataset/curves", response_model=CurveList)
def dataset_curves(request: Request, name: str, version: str) -> CurveList:
    """GPT · SoVITS 학습 로그의 스칼라 곡선. 학습 로그가 없으면 빈 목록이다."""
    voice_dir = _voice_dir(request, name)
    curves: list[CurveInfo] = []
    available = True
    for source, key in (("gpt", "gpt_logs"), ("sovits", "sovits_logs")):
        stage = STAGES[key]
        directory = stage.directory(voice_dir, validate_name(version, "version"))
        if not directory.is_dir():
            continue
        series = read_curves(source, directory)
        if series is None:
            available = False
            break
        curves.extend(CurveInfo(source=s.source, tag=s.tag, points=s.points) for s in series)
    return CurveList(available=available, curves=curves)


@router.get("/voices/{name}/dataset/{stage}/files", response_model=FileList)
def dataset_files(
    request: Request, name: str, stage: str, version: str | None = None,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=_MAX_PAGE),
) -> FileList:
    """단계 폴더의 파일 목록 한 페이지. 오디오는 길이를 함께 준다."""
    spec = _stage(stage)
    directory = _stage_dir(_voice_dir(request, name), spec, version)
    total, entries = list_files(directory, spec.recursive, offset, limit)
    return FileList(total=total, items=[FileInfo(**vars(e)) for e in entries])


@router.get("/voices/{name}/dataset/{stage}/files/{file_path:path}")
def dataset_file(
    request: Request, name: str, stage: str, file_path: str, version: str | None = None,
) -> FileResponse:
    """단계 폴더의 오디오 · 텍스트 파일 내용. 그 밖의 종류(가중치 · 특징 파일)는 내주지 않는다."""
    spec = _stage(stage)
    directory = _stage_dir(_voice_dir(request, name), spec, version)
    target = directory / file_path
    if not is_within(str(directory), str(target)) or not target.is_file():
        raise HTTPException(404, detail=f"파일 없음: {file_path}")
    suffix = target.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return FileResponse(target)
    if suffix in TEXT_SUFFIXES:
        return FileResponse(target, media_type="text/plain; charset=utf-8")
    raise HTTPException(415, detail="열람할 수 없는 파일 종류입니다")
