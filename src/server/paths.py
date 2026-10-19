"""요청에서 들어온 이름/경로가 허용 범위를 벗어나지 않는지 검증한다."""
from __future__ import annotations

import os

from fastapi import HTTPException


def validate_name(value: str, what: str = "이름") -> str:
    """경로 구성요소로 쓸 수 있는 단일 이름인지 검증한다 (구분자, '.', '..' 거부)."""
    if (
        not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise HTTPException(400, detail=f"유효하지 않은 {what}: {value!r}")
    return value


def is_within(base: str, path: str) -> bool:
    """path 가 symlink 를 해석한 뒤에도 base 아래에 있는지."""
    base_real = os.path.realpath(base)
    path_real = os.path.realpath(path)
    return os.path.commonpath([base_real, path_real]) == base_real
