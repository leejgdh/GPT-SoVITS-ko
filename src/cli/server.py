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

    from _setup_paths import setup_gpt_sovits_paths
    from src.config.config import Config, load_config

    # GPT_SoVITS 내부 모듈 import를 위한 경로 설정 (메인 스레드에서 실행)
    setup_gpt_sovits_paths()

    path = Path(config_path)
    if path.exists():
        config = load_config(path)
        os.environ["TTS_SERVICE_CONFIG"] = str(config_path)
    else:
        config = Config()

    host = config.service.host
    port = config.service.port

    # pretrained 모델 존재 여부 체크
    from src.config.config import pretrained_gpt_name
    project_root = Path(__file__).resolve().parents[2]
    sample_model = project_root / pretrained_gpt_name.get("v2Pro", "")
    if not sample_model.exists():
        logger.info(
            "pretrained 모델 미설치 — 서버를 시작하지 않습니다.\n"
            "  파이프라인 완료 후 'python main.py serve'로 실행하세요.",
        )
        return

    def _run_server():
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
