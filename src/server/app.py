from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.config import Config, load_config
from src.metrics import http_duration_seconds, http_requests_total, process_up
from src.server.context import ServiceContext  # noqa: F401 (sys.path 설정 포함)
from src.server.routers import all_routers


class _RequestIdMiddleware(BaseHTTPMiddleware):
    """요청별 request_id를 생성하고 로그 컨텍스트에 설정하는 미들웨어."""

    async def dispatch(
        self, request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        rid = request.headers.get("X-Request-ID", uuid4().hex[:8])
        with logger.contextualize(request_id=rid):
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response


class _MetricsMiddleware(BaseHTTPMiddleware):
    """모든 HTTP 요청에 대해 requests_total / duration 자동 기록.

    path 라벨은 라우트 템플릿(`/voices/{name}/emotions/{emotion}`) 으로 설정해
    기수 높은 path segment 로 시계열이 폭발하지 않도록 한다. /metrics 는
    self-probe 방지로 제외.
    """

    async def dispatch(
        self, request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        raw_path = request.url.path
        if raw_path.startswith("/metrics"):
            return await call_next(request)

        t0 = time.monotonic()
        try:
            response = await call_next(request)
            result = "ok" if response.status_code < 500 else "error"
            return response
        except Exception:
            result = "error"
            raise
        finally:
            elapsed = time.monotonic() - t0
            route = request.scope.get("route")
            path = getattr(route, "path", raw_path) if route is not None else raw_path
            labels = {"method": request.method, "path": path, "result": result}
            http_requests_total.labels(**labels).inc()
            http_duration_seconds.labels(**labels).observe(elapsed)


def create_app() -> FastAPI:
    """FastAPI 앱을 생성한다. uvicorn factory 모드에서 호출."""
    config_path = Path(os.environ.get("TTS_SERVICE_CONFIG", "config.yaml"))
    config = load_config(config_path) if config_path.exists() else Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = ServiceContext.create(config)
        ctx.warmup()
        app.state.context = ctx
        process_up.set(1)
        logger.info(
            "tts-service 시작 (host={}, port={}, voices={})",
            config.service.host,
            config.service.port,
            len(ctx.voices),
        )
        try:
            yield
        finally:
            process_up.set(0)
            ctx.close()
            logger.info("tts-service 종료")

    app = FastAPI(
        title="GPT-SoVITS TTS Service",
        lifespan=lifespan,
        docs_url="/swagger",
        redoc_url=None,
    )
    # 순서 주의: outer → inner. RequestId 먼저, Metrics 가 실핸들러 가까이.
    app.add_middleware(_MetricsMiddleware)
    app.add_middleware(_RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in all_routers:
        app.include_router(r)
    # Prometheus 메트릭 엔드포인트.
    app.mount("/metrics", make_asgi_app())
    return app
