"""job 실행기 — 대기열의 job 을 하나씩 subprocess 로 돌리고 상태 · 로그를 남긴다."""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import time
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from loguru import logger

from src.jobs.kinds import KINDS, JobKind, build_argv
from src.jobs.models import CANCELED, FAILED, INTERRUPTED, QUEUED, RUNNING, SUCCEEDED, Job, JobStep
from src.jobs.store import JobStore
from src.metrics import job_duration_seconds, jobs_total

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP_START = re.compile(r">>> (?P<label>.+) 시작$")
_STEP_DONE = re.compile(r"<<< (?P<label>.+) 완료 \((?P<seconds>\d+)초\)$")
_STEP_FAILED = re.compile(r"<<< (?P<label>.+) 실패 \(exit=(?P<code>-?\d+)\)$")
_READ_BYTES = 65536
_TERMINATE_GRACE_SECONDS = 10.0
_RESTART_ERROR = "서버가 다시 시작되어 중단됐습니다"


class ModelHost(Protocol):
    """job 실행 앞뒤로 모델 상태를 맞추는 쪽이 갖춰야 하는 동작."""

    async def unload_models(self) -> bool:
        """VRAM 의 모델을 내린다."""
        ...

    async def reload_voices(self) -> int:
        """voice 폴더를 다시 읽는다."""
        ...


class JobRunner:
    """단일 worker 로 job 을 순서대로 실행한다. GPU 를 한 job 이 쓰도록 동시에 하나만 돌린다."""

    def __init__(self, store: JobStore, host: ModelHost, project_root: Path, voices_dir: str) -> None:
        self._store = store
        self._host = host
        self._root = project_root
        self._voices_dir = voices_dir
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._running_id: str | None = None
        self._cancel_requested: set[str] = set()
        self._finished: dict[str, asyncio.Event] = {}
        self._stopping = False

    # ── 생명주기 ─────────────────────────────────────────────────────

    def start(self) -> None:
        """저장된 job 을 읽고 worker 를 시작한다. 이전 실행에서 끝나지 못한 job 은 중단 처리한다."""
        for job in self._store.load_all():
            if job.active:
                self._finish(job, INTERRUPTED, error=_RESTART_ERROR)
            self._jobs[job.id] = job
        self._worker = asyncio.create_task(self._work())

    async def stop(self) -> None:
        """실행 중인 job 을 중단하고 worker 를 멈춘다."""
        self._stopping = True
        if self._process is not None:
            await self._terminate(self._process)
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    # ── 조회 · 등록 · 취소 ───────────────────────────────────────────

    @property
    def store(self) -> JobStore:
        return self._store

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self, voice: str | None, status: str | None) -> list[Job]:
        """최근 등록순(내림차순) 목록. ``voice`` · ``status`` 로 거른다."""
        jobs = [
            j for j in self._jobs.values()
            if (voice is None or j.voice == voice) and (status is None or j.status == status)
        ]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def is_voice_busy(self, voice: str) -> bool:
        """그 voice 의 대기 중이거나 실행 중인 job 이 있는가."""
        return any(j.voice == voice and j.active for j in self._jobs.values())

    def submit(self, voice: str, kind: str, params: dict) -> Job:
        """job 을 대기열에 올린다. ``kind`` 와 ``params`` 는 검증을 마친 값이어야 한다."""
        job = Job(
            id=uuid4().hex[:12], voice=voice, kind=kind, params=params,
            status=QUEUED, created_at=time.time(),
        )
        self._jobs[job.id] = job
        self._store.save(job)
        self._queue.put_nowait(job.id)
        logger.info("job 등록: id={} voice={} kind={}", job.id, voice, kind)
        return job

    async def cancel(self, job_id: str) -> Job | None:
        """대기 중인 job 은 바로, 실행 중인 job 은 프로세스를 종료해 취소한다. 종료 상태가 기록된 뒤 돌려준다."""
        job = self._jobs.get(job_id)
        if job is None or not job.active:
            return job
        self._cancel_requested.add(job_id)
        if job.status == QUEUED:
            self._finish(job, CANCELED)
        elif self._process is not None and self._running_id == job_id:
            finished = self._finished.setdefault(job_id, asyncio.Event())
            await self._terminate(self._process)
            await finished.wait()
        return job

    # ── 실행 ─────────────────────────────────────────────────────────

    async def _work(self) -> None:
        while True:
            job = self._jobs[await self._queue.get()]
            if job.status != QUEUED:
                continue
            try:
                await self._run(job)
            except Exception:
                logger.exception("job 실행 실패: {}", job.id)
                self._finish(job, FAILED, error="실행 중 오류가 발생했습니다")

    async def _run(self, job: Job) -> None:
        kind = KINDS[job.kind]
        job.status = RUNNING
        job.started_at = time.time()
        self._store.save(job)
        self._running_id = job.id
        try:
            await self._host.unload_models()
            process = await self._spawn(job, kind)
            self._process = process
            logger.info("job 시작: id={} pid={}", job.id, process.pid)
            await self._pump_output(job, process)
            code = await process.wait()
        finally:
            self._process = None
            self._running_id = None

        if self._stopping:
            self._finish(job, INTERRUPTED, error=_RESTART_ERROR, exit_code=code)
        elif job.id in self._cancel_requested:
            self._finish(job, CANCELED, exit_code=code)
        elif code == 0:
            self._finish(job, SUCCEEDED, exit_code=code)
        else:
            self._finish(job, FAILED, exit_code=code, error=f"종료 코드 {code}")
        await self._refresh_voices()

    async def _spawn(self, job: Job, kind: JobKind) -> asyncio.subprocess.Process:
        voice_dir = self._voice_dir(job.voice)
        params = kind.params.model_validate(job.params)
        argv = build_argv(kind, voice_dir, params)
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )

    def _voice_dir(self, voice: str) -> str:
        return os.path.join(self._voices_dir, voice)

    async def _pump_output(self, job: Job, process: asyncio.subprocess.Process) -> None:
        """출력을 로그 파일에 쌓고, 줄 단위로 단계 표식을 읽어 상태에 반영한다."""
        assert process.stdout is not None
        partial = ""
        with open(self._store.log_path(job.id), "a", encoding="utf-8") as log:
            while chunk := await process.stdout.read(_READ_BYTES):
                text = _ANSI.sub("", chunk.decode("utf-8", errors="replace")).replace("\r", "\n")
                log.write(text)
                log.flush()
                *lines, partial = (partial + text).split("\n")
                if any([self._track_step(job, line) for line in lines]):
                    self._store.save(job)

    @staticmethod
    def _track_step(job: Job, line: str) -> bool:
        """단계 표식 줄이면 job.steps 를 갱신하고 True."""
        line = line.rstrip()
        if m := _STEP_START.search(line):
            job.steps.append(JobStep(label=m["label"], status=RUNNING))
            return True
        for pattern, status in ((_STEP_DONE, SUCCEEDED), (_STEP_FAILED, FAILED)):
            if m := pattern.search(line):
                step = next((s for s in reversed(job.steps) if s.label == m["label"] and s.status == RUNNING), None)
                if step is None:
                    step = JobStep(label=m["label"], status=status)
                    job.steps.append(step)
                step.status = status
                step.seconds = float(m["seconds"]) if status == SUCCEEDED else None
                return True
        return False

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """프로세스 그룹에 종료를 보내고, 유예 시간이 지나도 남아 있으면 강제 종료한다."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), _TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    async def _refresh_voices(self) -> None:
        try:
            await self._host.reload_voices()
        except Exception:
            logger.exception("job 종료 후 voice 재스캔 실패")

    def _finish(self, job: Job, status: str, *, error: str | None = None, exit_code: int | None = None) -> None:
        job.status = status
        job.finished_at = time.time()
        job.error = error
        job.exit_code = exit_code
        self._store.save(job)
        self._cancel_requested.discard(job.id)
        if (finished := self._finished.pop(job.id, None)) is not None:
            finished.set()
        if job.started_at is not None:
            jobs_total.labels(kind=job.kind, result=status).inc()
            job_duration_seconds.labels(kind=job.kind, result=status).observe(job.finished_at - job.started_at)
        logger.info("job 종료: id={} status={} exit={}", job.id, status, exit_code)
