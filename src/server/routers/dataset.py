"""voice 별 학습 데이터(단계별 산출물) 열람 API — 읽기 전용."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.dataset.files import AUDIO_SUFFIXES, TEXT_SUFFIXES, list_files, summarize
from src.dataset.stages import STAGES, voice_versions
from src.dataset.tables import read_curves, read_semantic_rows, read_text_rows
from src.server.paths import is_within, validate_name
from src.server.stage_access import stage_dir_of, stage_of, voice_dir_of

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
    accepts: list[str]


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
# 엔드포인트
# ---------------------------------------------------------------------------

@router.get("/voices/{name}/dataset", response_model=DatasetSummary)
def dataset_summary(request: Request, name: str, version: str | None = None) -> DatasetSummary:
    """모든 단계의 폴더 존재 여부 · 파일 수 · 용량 · 올릴 수 있는 확장자를 반환한다.

    ``version`` 을 주면 아직 폴더가 없는 그 버전의 단계도 함께 싣는다. ``versions`` 는 폴더가 있는 버전만이다.
    """
    voice_dir = voice_dir_of(request, name)
    versions = voice_versions(voice_dir)
    shown = sorted({*versions, validate_name(version, "version")} if version else versions)
    infos: list[StageInfo] = []
    for stage in STAGES.values():
        for stage_version in (shown if stage.versioned else [None]):
            result = summarize(stage.directory(voice_dir, stage_version), stage.recursive)
            infos.append(StageInfo(
                key=stage.key, version=stage_version,
                exists=result.exists, files=result.files, bytes=result.bytes,
                accepts=sorted(stage.accepts),
            ))
    return DatasetSummary(voice=name, versions=versions, stages=infos)


@router.get("/voices/{name}/dataset/text", response_model=TextRowList)
def dataset_text(
    request: Request, name: str, version: str,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=_MAX_PAGE),
) -> TextRowList:
    """step2 음소 파일(name2text.txt)의 한 페이지."""
    directory = stage_dir_of(voice_dir_of(request, name), STAGES["step2"], version)
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
    directory = stage_dir_of(voice_dir_of(request, name), STAGES["step2"], version)
    path = directory / "name2semantic.tsv"
    if not path.is_file():
        raise HTTPException(404, detail="semantic 파일이 없습니다")
    total, rows = read_semantic_rows(path, offset, limit)
    return SemanticRowList(total=total, items=[SemanticRowInfo(**vars(r)) for r in rows])


@router.get("/voices/{name}/dataset/curves", response_model=CurveList)
def dataset_curves(request: Request, name: str, version: str) -> CurveList:
    """GPT · SoVITS 학습 로그의 스칼라 곡선. 학습 로그가 없으면 빈 목록이다."""
    voice_dir = voice_dir_of(request, name)
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
    spec = stage_of(stage)
    directory = stage_dir_of(voice_dir_of(request, name), spec, version)
    total, entries = list_files(directory, spec.recursive, offset, limit)
    return FileList(total=total, items=[FileInfo(**vars(e)) for e in entries])


@router.get("/voices/{name}/dataset/{stage}/files/{file_path:path}")
def dataset_file(
    request: Request, name: str, stage: str, file_path: str, version: str | None = None,
) -> FileResponse:
    """단계 폴더의 오디오 · 텍스트 파일 내용. 그 밖의 종류(가중치 · 특징 파일)는 내주지 않는다."""
    spec = stage_of(stage)
    directory = stage_dir_of(voice_dir_of(request, name), spec, version)
    target = directory / file_path
    if not is_within(str(directory), str(target)) or not target.is_file():
        raise HTTPException(404, detail=f"파일 없음: {file_path}")
    suffix = target.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return FileResponse(target)
    if suffix in TEXT_SUFFIXES:
        return FileResponse(target, media_type="text/plain; charset=utf-8")
    raise HTTPException(415, detail="열람할 수 없는 파일 종류입니다")
