from __future__ import annotations

import mimetypes
import os
import tempfile

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.server.context import ServiceContext

router = APIRouter()

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


# ---------------------------------------------------------------------------
# Request 모델
# ---------------------------------------------------------------------------

class LabelUpdateRequest(BaseModel):
    text: str
    lang: str | None = None


class LabelStateRequest(BaseModel):
    state: str  # pending, approved, rejected


# ---------------------------------------------------------------------------
# 라벨 유틸
# ---------------------------------------------------------------------------

_LABEL_FILE = os.path.join("step1", "04_asr", "03_vocal.list")

_VALID_STATES = {"pending", "approved", "rejected"}


def get_label_path(voices_dir: str, voice_name: str) -> str:
    """voice의 ASR 라벨 파일 경로를 반환한다."""
    return os.path.join(voices_dir, voice_name, _LABEL_FILE)


def read_labels(label_path: str) -> list[dict]:
    """라벨 파일을 파싱하여 리스트로 반환한다.

    각 라인 형식: audio_path|category|lang| text|state
    """
    labels: list[dict] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            state = parts[4].strip() if len(parts) >= 5 else "pending"
            if state not in _VALID_STATES:
                state = "pending"
            labels.append({
                "path": parts[0],
                "category": parts[1],
                "lang": parts[2],
                "text": parts[3].strip(),
                "state": state,
            })
    return labels


def write_labels(label_path: str, labels: list[dict]) -> None:
    """라벨 리스트를 파일에 원자적으로 쓴다."""
    dir_name = os.path.dirname(label_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for lb in labels:
                state = lb.get("state", "pending")
                f.write(
                    f"{lb['path']}|{lb['category']}|{lb['lang']}"
                    f"| {lb['text']}|{state}\n",
                )
        os.replace(tmp_path, label_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


@router.get("/voices/{name}/labels")
async def list_labels(request: Request, name: str):
    """voice의 ASR 라벨 목록을 반환한다."""
    ctx = _get_context(request)
    label_path = get_label_path(ctx.config.voices_dir, name)

    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for lb in labels:
        counts[lb["state"]] = counts.get(lb["state"], 0) + 1

    return JSONResponse(content={
        "voice": name,
        "total": len(labels),
        "counts": counts,
        "labels": [
            {
                "index": i,
                "audio_file": os.path.basename(lb["path"]),
                "lang": lb["lang"],
                "text": lb["text"],
                "state": lb["state"],
            }
            for i, lb in enumerate(labels)
        ],
    })


@router.get("/voices/{name}/labels/{index}/audio")
async def get_label_audio(request: Request, name: str, index: int):
    """라벨에 해당하는 오디오 파일을 스트리밍한다."""
    ctx = _get_context(request)
    label_path = get_label_path(ctx.config.voices_dir, name)

    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    if index < 0 or index >= len(labels):
        return JSONResponse(
            status_code=404,
            content={"message": f"인덱스 범위 초과: {index} (총 {len(labels)}개)"},
        )

    audio_path = labels[index]["path"]
    # 상대경로이면 프로젝트 루트 기준으로 변환
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(_PROJECT_ROOT, audio_path)

    if not os.path.isfile(audio_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"오디오 파일을 찾을 수 없습니다: {audio_path}"},
        )

    media_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    return FileResponse(audio_path, media_type=media_type)


@router.put("/voices/{name}/labels/{index}")
async def update_label(
    request: Request, name: str, index: int, body: LabelUpdateRequest,
):
    """라벨의 텍스트(및 언어)를 수정한다."""
    ctx = _get_context(request)
    label_path = get_label_path(ctx.config.voices_dir, name)

    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    if index < 0 or index >= len(labels):
        return JSONResponse(
            status_code=404,
            content={"message": f"인덱스 범위 초과: {index} (총 {len(labels)}개)"},
        )

    labels[index]["text"] = body.text
    if body.lang is not None:
        labels[index]["lang"] = body.lang

    write_labels(label_path, labels)
    logger.info(
        "라벨 수정: voice={}, index={}, text='{}'",
        name, index, body.text,
    )

    return JSONResponse(content={
        "message": "수정 완료",
        "index": index,
        "text": body.text,
        "lang": labels[index]["lang"],
    })


@router.patch("/voices/{name}/labels/{index}/state")
async def update_label_state(
    request: Request, name: str, index: int, body: LabelStateRequest,
):
    """라벨의 상태를 변경한다 (pending/approved/rejected)."""
    if body.state not in _VALID_STATES:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"유효하지 않은 상태: {body.state}"
                f" (허용: {', '.join(sorted(_VALID_STATES))})",
            },
        )

    ctx = _get_context(request)
    label_path = get_label_path(ctx.config.voices_dir, name)

    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    if index < 0 or index >= len(labels):
        return JSONResponse(
            status_code=404,
            content={"message": f"인덱스 범위 초과: {index} (총 {len(labels)}개)"},
        )

    old_state = labels[index]["state"]
    labels[index]["state"] = body.state
    write_labels(label_path, labels)
    logger.info(
        "라벨 상태 변경: voice={}, index={}, {} → {}",
        name, index, old_state, body.state,
    )

    return JSONResponse(content={
        "message": "상태 변경 완료",
        "index": index,
        "old_state": old_state,
        "new_state": body.state,
    })


@router.delete("/voices/{name}/labels/{index}")
async def delete_label(request: Request, name: str, index: int):
    """라벨 항목과 해당 오디오 파일을 삭제한다."""
    ctx = _get_context(request)
    label_path = get_label_path(ctx.config.voices_dir, name)

    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    if index < 0 or index >= len(labels):
        return JSONResponse(
            status_code=404,
            content={"message": f"인덱스 범위 초과: {index} (총 {len(labels)}개)"},
        )

    removed = labels.pop(index)

    # 오디오 파일 삭제
    audio_path = removed["path"]
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(_PROJECT_ROOT, audio_path)

    audio_deleted = False
    if os.path.isfile(audio_path):
        os.remove(audio_path)
        audio_deleted = True

    write_labels(label_path, labels)
    logger.info(
        "라벨 삭제: voice={}, index={}, audio_deleted={}",
        name, index, audio_deleted,
    )

    return JSONResponse(content={
        "message": "삭제 완료",
        "removed_text": removed["text"],
        "audio_deleted": audio_deleted,
        "remaining": len(labels),
    })
