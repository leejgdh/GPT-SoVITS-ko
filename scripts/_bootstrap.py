"""scripts/ 공통 부트스트랩.

각 스크립트 최상단에서 import하여 sys.path를 설정한다.
경로 설정은 프로젝트 루트의 _setup_paths.py에 위임한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_paths() -> Path:
    """프로젝트 루트를 sys.path에 추가하고 반환한다.

    경로 설정 목록은 _setup_paths.setup_gpt_sovits_paths()에서 관리한다.
    """
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from _setup_paths import setup_gpt_sovits_paths
    setup_gpt_sovits_paths()
    return project_root


def filter_label_lines(lines: list[str]) -> list[str]:
    """라벨 상태에 따라 학습에 사용할 라인만 필터링한다.

    - approved가 하나라도 있으면 → approved만 반환
    - approved가 없으면 → pending만 반환 (rejected 제외)
    """
    approved: list[str] = []
    pending: list[str] = []

    for line in lines:
        parts = line.split("|")
        state = parts[4].strip() if len(parts) >= 5 else "pending"
        if state == "approved":
            approved.append(line)
        elif state != "rejected":
            pending.append(line)

    if approved:
        logger.info(
            "라벨 필터: approved {} / pending {} / 전체 {} → approved만 사용",
            len(approved), len(pending), len(lines),
        )
        return approved

    logger.info(
        "라벨 필터: approved 없음 / pending {} / 전체 {} → pending만 사용",
        len(pending), len(lines),
    )
    return pending


def parse_label_line(line: str) -> tuple[str, str, str, str]:
    """라벨 라인을 파싱하여 (wav_name, spk_name, language, text)를 반환한다."""
    parts = line.split("|")
    return parts[0], parts[1], parts[2], parts[3] if len(parts) >= 4 else ""
