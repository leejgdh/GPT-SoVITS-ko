"""학습 파이프라인 job API — 등록 · 조회 · 로그 · 취소."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError

from src.jobs.kinds import KINDS
from src.jobs.models import Job
from src.jobs.runner import JobRunner
from src.server.context import ServiceContext
from src.server.paths import is_within, validate_name

router = APIRouter(tags=["jobs"])

_LOG_CHUNK_BYTES = 256 * 1024


# ---------------------------------------------------------------------------
# 요청 · 응답 모델
# ---------------------------------------------------------------------------

class JobCreateRequest(BaseModel):
    voice: str
    kind: str
    params: dict[str, Any]


class JobStepInfo(BaseModel):
    label: str
    status: str
    seconds: float | None


class JobInfo(BaseModel):
    id: str
    voice: str
    kind: str
    params: dict[str, Any]
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    exit_code: int | None
    error: str | None
    steps: list[JobStepInfo]


class JobList(BaseModel):
    items: list[JobInfo]


class JobLogChunk(BaseModel):
    text: str
    offset: int
    done: bool


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _runner(request: Request) -> JobRunner:
    return request.app.state.jobs


def _info(job: Job) -> JobInfo:
    return JobInfo(
        id=job.id, voice=job.voice, kind=job.kind, params=job.params, status=job.status,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        exit_code=job.exit_code, error=job.error,
        steps=[JobStepInfo(label=s.label, status=s.status, seconds=s.seconds) for s in job.steps],
    )


def _require(request: Request, job_id: str) -> Job:
    job = _runner(request).get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"job '{job_id}' 이(가) 없습니다")
    return job


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.post("/jobs", status_code=202, response_model=JobInfo)
async def create_job(request: Request, body: JobCreateRequest) -> JobInfo:
    """job 을 대기열에 올린다. 한 번에 하나씩 순서대로 실행된다."""
    ctx: ServiceContext = request.app.state.context
    kind = KINDS.get(body.kind)
    if kind is None:
        raise HTTPException(400, detail=f"알 수 없는 kind: {body.kind}. 가능한 값: {sorted(KINDS)}")

    voice_dir = Path(ctx.config.voices_dir) / validate_name(body.voice, "voice 이름")
    if not voice_dir.is_dir():
        raise HTTPException(404, detail=f"voice '{body.voice}' 폴더가 없습니다")

    try:
        params = kind.params.model_validate(body.params)
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_url=False, include_context=False)) from exc

    for field in kind.voice_paths:
        value = getattr(params, field)
        if value is not None:
            target = voice_dir / value
            if not is_within(str(voice_dir), str(target)) or not target.is_file():
                raise HTTPException(400, detail=f"{field} 파일이 voice 폴더 안에 없습니다: {value}")

    job = _runner(request).submit(body.voice, body.kind, params.model_dump(mode="json"))
    return _info(job)


@router.get("/jobs", response_model=JobList)
async def list_jobs(request: Request, voice: str | None = None, status: str | None = None) -> JobList:
    """job 목록. 최근 등록순이다."""
    return JobList(items=[_info(j) for j in _runner(request).list_jobs(voice, status)])


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(request: Request, job_id: str) -> JobInfo:
    """job 의 상태와 단계별 결과."""
    return _info(_require(request, job_id))


@router.get("/jobs/{job_id}/log", response_model=JobLogChunk)
async def get_job_log(request: Request, job_id: str, offset: int = Query(0, ge=0)) -> JobLogChunk:
    """로그를 ``offset`` 바이트부터 읽는다. 응답의 ``offset`` 을 다음 요청에 넘기면 이어 읽는다."""
    job = _require(request, job_id)
    store = _runner(request).store
    text, next_offset = store.read_log(job_id, offset, _LOG_CHUNK_BYTES)
    log_path = store.log_path(job_id)
    size = log_path.stat().st_size if log_path.is_file() else 0
    return JobLogChunk(text=text, offset=next_offset, done=not job.active and next_offset >= size)


@router.post("/jobs/{job_id}/cancel", response_model=JobInfo)
async def cancel_job(request: Request, job_id: str) -> JobInfo:
    """대기 중이면 대기열에서 빼고, 실행 중이면 프로세스를 종료한다. 이미 끝난 job 은 그대로 돌려준다."""
    _require(request, job_id)
    job = await _runner(request).cancel(job_id)
    assert job is not None
    return _info(job)
