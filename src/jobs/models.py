"""job 상태 모델."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELED = "canceled"
INTERRUPTED = "interrupted"

ACTIVE_STATUSES = frozenset({QUEUED, RUNNING})


@dataclass
class JobStep:
    """job 안에서 실행된 한 단계. ``seconds`` 는 끝난 단계에만 있다."""

    label: str
    status: str
    seconds: float | None = None


@dataclass
class Job:
    """등록된 job 하나. 시각은 UTC epoch 초다."""

    id: str
    voice: str
    kind: str
    params: dict[str, Any]
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    steps: list[JobStep] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES
