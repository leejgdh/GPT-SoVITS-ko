"""서버 관리 — 백그라운드 시작/대기."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from loguru import logger


_server_thread: threading.Thread | None = None


def start_server_background(config_path: str = "conf.yaml") -> None:
    """서버를 백그라운드 스레드로 시작한다. 이미 실행 중이면 무시."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        return

    import uvicorn

    from src.config.config import Config, load_config

    path = Path(config_path)
    if path.exists():
        config = load_config(path)
        os.environ["TTS_SERVICE_CONFIG"] = str(config_path)
    else:
        config = Config()

    host = config.service.host
    port = config.service.port

    def _run_server():
        from _setup_paths import setup_gpt_sovits_paths
        setup_gpt_sovits_paths()
        uvicorn.run(
            "src.server.app:create_app",
            factory=True,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
    logger.info("서버 시작 (백그라운드): http://{}:{}", host, port)


def wait_for_server() -> None:
    """서버가 실행 중이면 Ctrl+C까지 대기한다."""
    if _server_thread is None or not _server_thread.is_alive():
        return
    logger.info("서버 실행 중 — Ctrl+C로 종료")
    try:
        while _server_thread.is_alive():
            _server_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("종료 요청")
