from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.config import Config, load_config
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


def create_app() -> FastAPI:
    """FastAPI 앱을 생성한다. uvicorn factory 모드에서 호출."""
    config_path = Path(os.environ.get("TTS_SERVICE_CONFIG", "conf.yaml"))
    config = load_config(config_path) if config_path.exists() else Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = ServiceContext.create(config)
        ctx.warmup()
        app.state.context = ctx
        logger.info(
            "tts-service 시작 (host={}, port={}, voices={})",
            config.service.host,
            config.service.port,
            len(ctx.voices),
        )
        yield
        ctx.close()
        logger.info("tts-service 종료")

    app = FastAPI(
        title="GPT-SoVITS TTS Service",
        lifespan=lifespan,
        docs_url="/swagger",
        redoc_url=None,
    )
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
    return app
