"""유휴 시간이 지난 모델을 내리는 백그라운드 작업."""
from __future__ import annotations

import asyncio
import contextlib

from loguru import logger

from src.server.context import ServiceContext

_POLL_SECONDS = 15.0


class IdleUnloader:
    """마지막 합성 이후 ``idle_seconds`` 가 지나면 ``ServiceContext`` 의 모델을 내린다."""

    def __init__(self, ctx: ServiceContext, idle_seconds: int) -> None:
        self._ctx = ctx
        self._idle_seconds = idle_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """감시 작업을 시작한다."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """감시 작업을 멈추고 끝나길 기다린다."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(_POLL_SECONDS)
            try:
                if await self._ctx.unload_if_idle(self._idle_seconds):
                    logger.info("유휴 {}초 경과 — 모델 unload", self._idle_seconds)
            except Exception:
                logger.exception("유휴 모델 unload 실패")
