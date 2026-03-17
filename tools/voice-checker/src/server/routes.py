"""라벨링 API 엔드포인트."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter()

_LABELER_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "labeler.html",
)


def _load_labels(labels_file: Path) -> list[dict]:
    if not labels_file.exists():
        return []
    with open(labels_file, encoding="utf-8") as f:
        return json.load(f).get("files", [])


def _save_labels(labels_file: Path, files: list[dict]) -> None:
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)


@router.get("/files")
async def list_files(request: Request):
    """파일 목록 + 라벨 + 통계를 반환한다."""
    labels = _load_labels(request.app.state.labels_file)
    counts = {"good": 0, "bad": 0, "unlabeled": 0}
    for entry in labels:
        lbl = entry.get("label", "unlabeled")
        counts[lbl] = counts.get(lbl, 0) + 1
    return {"files": labels, "counts": counts}


@router.get("/files/{name}/audio")
async def get_audio(name: str, request: Request):
    """오디오 파일을 스트리밍한다."""
    audio_path = request.app.state.data_dir / name
    if not audio_path.exists():
        raise HTTPException(404, detail=f"파일 없음: {name}")
    return FileResponse(audio_path, media_type="audio/wav")


class LabelUpdate(BaseModel):
    label: str


@router.patch("/files/{name}/label")
async def update_label(name: str, body: LabelUpdate, request: Request):
    """파일의 라벨을 변경한다."""
    if body.label not in ("good", "bad", "unlabeled"):
        raise HTTPException(400, detail="label은 good, bad, unlabeled 중 하나여야 합니다")

    labels = _load_labels(request.app.state.labels_file)
    found = False
    for entry in labels:
        if entry["name"] == name:
            entry["label"] = body.label
            found = True
            break

    if not found:
        raise HTTPException(404, detail=f"파일 없음: {name}")

    _save_labels(request.app.state.labels_file, labels)
    logger.info("{} -> {}", name, body.label)
    return {"name": name, "label": body.label}


@router.get("/labeler", response_class=HTMLResponse)
async def labeler_page():
    """라벨링 UI를 서빙한다."""
    with open(_LABELER_HTML, encoding="utf-8") as f:
        html = f.read()
    html = html.replace('value="http://localhost:9890"', 'value=""')
    html = html.replace(
        "document.addEventListener('DOMContentLoaded', init);",
        "document.addEventListener('DOMContentLoaded', () => {"
        " $('#server-url').value = window.location.origin;"
        " init();"
        " });",
    )
    return HTMLResponse(content=html)
