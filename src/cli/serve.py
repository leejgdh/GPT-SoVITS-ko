"""serve 커맨드."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from src.cli.logger import LOG_DIR, setup_logger


def cmd_serve(args: argparse.Namespace) -> None:
    """REST API 서버를 포그라운드로 실행한다."""
    import uvicorn

    from src.config.config import load_config
    from src.exceptions import ConfigError

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
    except ConfigError as e:
        print(f"설정 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)
    os.environ["TTS_SERVICE_CONFIG"] = str(config_path)

    log_level = "DEBUG" if args.verbose else config.log_level
    setup_logger("tts_service", level=log_level, log_dir=LOG_DIR)

    host = args.host or config.service.host
    port = args.port or config.service.port

    logger.info("GPT-SoVITS TTS Service 시작: {}:{}", host, port)

    uvicorn.run(
        "src.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False,
    )
