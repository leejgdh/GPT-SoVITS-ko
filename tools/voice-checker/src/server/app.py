"""FastAPI 애플리케이션 팩토리."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.config import Config, load_config
from src.server.routes import router


class _RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID", uuid4().hex[:8])
        with logger.contextualize(request_id=rid):
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response


def create_app() -> FastAPI:
    """uvicorn factory 모드에서 호출된다."""
    config_path = Path(os.environ.get("VOICE_CHECKER_CONFIG", "conf.yaml"))
    config = load_config(config_path) if config_path.exists() else Config()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    labels_file = data_dir / "labels.json"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.data_dir = data_dir
        app.state.labels_file = labels_file
        logger.info("Voice Checker 서버 시작")
        yield
        logger.info("Voice Checker 서버 종료")

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(_RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
