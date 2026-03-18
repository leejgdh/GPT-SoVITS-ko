"""로거 설정."""
from __future__ import annotations

import logging as _logging
import sys
from pathlib import Path

from loguru import logger

_CONSOLE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss,SSS} [{extra[request_id]}] "
    "[<level>{level}</level>] {name}: {message}"
)
_FILE_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss,SSS} [{extra[request_id]}] "
    "[{level}] {name}: {message}"
)

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def setup_logger(
    root_name: str,
    level: str = "INFO",
    log_dir: Path | None = LOG_DIR,
) -> None:
    """loguru 로거를 설정한다."""
    logger.remove()

    def _patcher(record: dict) -> None:
        name = record["name"] or ""
        if name.startswith("src."):
            name = name[4:]
        if name == "__main__":
            record["name"] = root_name
        elif not name.startswith(f"{root_name}."):
            record["name"] = f"{root_name}.{name}"

    logger.configure(extra={"request_id": "-"}, patcher=_patcher)
    logger.add(
        sys.stderr, format=_CONSOLE_FMT,
        level=level.upper(), colorize=True,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_dir / "gpt-sovits_{time:YYYY-MM-DD}.log"),
            format=_FILE_FMT, level="DEBUG",
            rotation="00:00", retention="30 days", encoding="utf-8",
        )

    class _InterceptHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            try:
                lvl: str | int = logger.level(record.levelname).name
            except ValueError:
                lvl = record.levelno
            frame, depth = _logging.currentframe(), 2
            while frame and frame.f_code.co_filename == _logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                lvl, record.getMessage(),
            )

    _logging.basicConfig(
        handlers=[_InterceptHandler()], level=0, force=True,
    )
