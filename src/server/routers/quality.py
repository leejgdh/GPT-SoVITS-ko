"""음질 라벨링 API — Voice Checker CNN 학습용 good/bad 분류."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
from pydantic import BaseModel

from src.server.context import ServiceContext

# ---------------------------------------------------------------------------
# 라우터 정의
# ---------------------------------------------------------------------------

api_router = APIRouter(prefix="/quality", tags=["quality"])
page_router = APIRouter(tags=["quality"])

_LABELER_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__),
    )))),
    "tools", "voice-checker", "tools", "labeler.html",
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


def _require_vc(request: Request) -> tuple[ServiceContext, Path, Path]:
    """Voice Checker 설정이 있는지 확인하고 (ctx, data_dir, labels_file)을 반환."""
    ctx = _get_context(request)
    if ctx.vc_data_dir is None or ctx.vc_labels_file is None:
        raise HTTPException(
            503, detail="Voice Checker가 설정되지 않았습니다. conf.yaml에 voice_checker 섹션을 추가하세요.",
        )
    return ctx, ctx.vc_data_dir, ctx.vc_labels_file


def _load_labels(labels_file: Path) -> list[dict]:
    if not labels_file.exists():
        return []
    with open(labels_file, encoding="utf-8") as f:
        return json.load(f).get("files", [])


def _save_labels(labels_file: Path, files: list[dict]) -> None:
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 페이지 서빙
# ---------------------------------------------------------------------------


@page_router.get("/quality-check", response_class=HTMLResponse)
async def quality_check_page():
    """음질 라벨링 UI를 서빙한다."""
    with open(_LABELER_HTML, encoding="utf-8") as f:
        html = f.read()
    html = html.replace('value="http://localhost:9880"', 'value=""')
    html = html.replace(
        "document.addEventListener('DOMContentLoaded', init);",
        "document.addEventListener('DOMContentLoaded', () => {"
        " $('#server-url').value = window.location.origin;"
        " init();"
        " });",
    )
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------


@api_router.get("/files")
async def list_files(request: Request):
    """파일 목록 + 라벨 + 통계를 반환한다."""
    _, _, labels_file = _require_vc(request)
    labels = _load_labels(labels_file)
    counts = {"good": 0, "bad": 0, "unlabeled": 0}
    for entry in labels:
        lbl = entry.get("label", "unlabeled")
        counts[lbl] = counts.get(lbl, 0) + 1
    return {"files": labels, "counts": counts}


@api_router.get("/files/{name}/audio")
async def get_audio(name: str, request: Request):
    """오디오 파일을 스트리밍한다."""
    _, data_dir, _ = _require_vc(request)
    audio_path = data_dir / name
    if not audio_path.exists():
        raise HTTPException(404, detail=f"파일 없음: {name}")
    return FileResponse(audio_path, media_type="audio/wav")


class LabelUpdate(BaseModel):
    label: str


@api_router.patch("/files/{name}/label")
async def update_label(name: str, body: LabelUpdate, request: Request):
    """파일의 라벨을 변경한다."""
    if body.label not in ("good", "bad", "unlabeled"):
        raise HTTPException(400, detail="label은 good, bad, unlabeled 중 하나여야 합니다")

    _, _, labels_file = _require_vc(request)
    labels = _load_labels(labels_file)
    found = False
    for entry in labels:
        if entry["name"] == name:
            entry["label"] = body.label
            found = True
            break

    if not found:
        raise HTTPException(404, detail=f"파일 없음: {name}")

    _save_labels(labels_file, labels)
    logger.info("{} -> {}", name, body.label)
    return {"name": name, "label": body.label}
