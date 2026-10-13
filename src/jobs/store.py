"""job 상태와 로그를 파일로 보관한다."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from src.jobs.models import Job, JobStep

_JOB_FILE = "job.json"
_LOG_FILE = "job.log"
_MAX_PARTIAL_CHAR_BYTES = 3


class JobStore:
    """``jobs_dir/{id}/job.json`` 과 ``job.log`` 를 읽고 쓴다."""

    def __init__(self, jobs_dir: Path) -> None:
        self._dir = jobs_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def log_path(self, job_id: str) -> Path:
        return self._dir / job_id / _LOG_FILE

    def save(self, job: Job) -> None:
        """job 상태를 원자적으로 쓴다."""
        directory = self._dir / job.id
        directory.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(job), f, ensure_ascii=False)
            os.replace(tmp, directory / _JOB_FILE)
        except BaseException:
            os.unlink(tmp)
            raise

    def load_all(self) -> list[Job]:
        """저장된 모든 job. 읽지 못하는 항목은 경고만 남기고 건너뛴다."""
        jobs: list[Job] = []
        for path in sorted(self._dir.glob(f"*/{_JOB_FILE}")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["steps"] = [JobStep(**s) for s in data["steps"]]
                jobs.append(Job(**data))
            except Exception:
                logger.exception("job 상태 읽기 실패: {}", path)
        return jobs

    def read_log(self, job_id: str, offset: int, max_bytes: int) -> tuple[str, int]:
        """로그를 ``offset`` 바이트부터 최대 ``max_bytes`` 읽는다. (본문, 다음 offset).

        UTF-8 문자가 chunk 경계에서 잘리면 그 문자는 다음 호출로 넘긴다.
        """
        path = self.log_path(job_id)
        if not path.is_file():
            return "", offset
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        cut = len(data)
        for trim in range(_MAX_PARTIAL_CHAR_BYTES + 1):
            try:
                data[:len(data) - trim].decode("utf-8")
            except UnicodeDecodeError:
                continue
            cut = len(data) - trim
            break
        return data[:cut].decode("utf-8", errors="replace"), offset + cut
