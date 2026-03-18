"""Voice Checker API — CNN 모델 학습용 오디오 라벨링 (good/bad).

오디오 파일은 data/voice/{voice}/step1/03_vocal/ 에서 직접 읽는다.
라벨은 data/voice/{voice}/quality_labels.json 에 저장한다.
"""
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

api_router = APIRouter(prefix="/voice-checker", tags=["voice-checker"])
page_router = APIRouter(tags=["voice-checker"])

_LABELER_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__),
    )))),
    "tools", "voice-checker", "tools", "labeler.html",
)

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
_LABELS_FILENAME = "quality_labels.json"


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


def _require_vc(request: Request) -> ServiceContext:
    ctx = _get_context(request)
    if ctx.config.voice_checker is None:
        raise HTTPException(
            503, detail="Voice Checker가 설정되지 않았습니다. conf.yaml에 voice_checker 섹션을 추가하세요.",
        )
    return ctx


def _get_vocal_dir(ctx: ServiceContext, voice: str) -> Path:
    """voice의 step1/03_vocal 디렉토리 경로를 반환한다."""
    vocal_dir = Path(ctx.config.voices_dir) / voice / "step1" / "03_vocal"
    if not vocal_dir.is_dir():
        raise HTTPException(404, detail=f"'{voice}' 의 step1/03_vocal 디렉토리가 없습니다")
    return vocal_dir


def _get_labels_path(ctx: ServiceContext, voice: str) -> Path:
    return Path(ctx.config.voices_dir) / voice / _LABELS_FILENAME


def _load_labels(labels_path: Path) -> dict[str, str]:
    """labels를 {filename: label} dict로 반환한다."""
    if not labels_path.exists():
        return {}
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["name"]: entry["label"] for entry in data.get("files", [])}


def _save_labels(labels_path: Path, labels: dict[str, str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    files = [{"name": name, "label": label} for name, label in sorted(labels.items())]
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)


def _scan_audio_files(vocal_dir: Path, labels: dict[str, str]) -> list[dict]:
    """vocal_dir의 오디오 파일을 스캔하고 라벨과 병합한다."""
    result = []
    for f in sorted(vocal_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in _AUDIO_EXTS:
            result.append({
                "name": f.name,
                "label": labels.get(f.name, "unlabeled"),
            })
    return result


# ---------------------------------------------------------------------------
# 페이지 서빙
# ---------------------------------------------------------------------------


@page_router.get("/voice-checker/labeling", response_class=HTMLResponse)
async def voice_checker_labeling_page(request: Request):
    """Voice Checker CNN 학습용 라벨링 UI를 서빙한다."""
    _require_vc(request)
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


@api_router.get("/{voice}/files")
async def list_files(voice: str, request: Request):
    """voice의 step1/03_vocal 오디오 파일 목록 + 라벨 + 통계를 반환한다."""
    ctx = _require_vc(request)
    vocal_dir = _get_vocal_dir(ctx, voice)
    labels = _load_labels(_get_labels_path(ctx, voice))
    files = _scan_audio_files(vocal_dir, labels)

    counts = {"good": 0, "bad": 0, "unlabeled": 0}
    for entry in files:
        counts[entry["label"]] = counts.get(entry["label"], 0) + 1

    return {"files": files, "counts": counts}


@api_router.get("/{voice}/files/{name}/audio")
async def get_audio(voice: str, name: str, request: Request):
    """오디오 파일을 스트리밍한다."""
    ctx = _require_vc(request)
    vocal_dir = _get_vocal_dir(ctx, voice)
    audio_path = vocal_dir / name
    if not audio_path.exists():
        raise HTTPException(404, detail=f"파일 없음: {name}")
    return FileResponse(audio_path, media_type="audio/wav")


class LabelUpdate(BaseModel):
    label: str


@api_router.patch("/{voice}/files/{name}/label")
async def update_label(voice: str, name: str, body: LabelUpdate, request: Request):
    """파일의 라벨을 변경한다."""
    if body.label not in ("good", "bad", "unlabeled"):
        raise HTTPException(400, detail="label은 good, bad, unlabeled 중 하나여야 합니다")

    ctx = _require_vc(request)
    vocal_dir = _get_vocal_dir(ctx, voice)
    if not (vocal_dir / name).exists():
        raise HTTPException(404, detail=f"파일 없음: {name}")

    labels = _load_labels(_get_labels_path(ctx, voice))
    labels[name] = body.label
    _save_labels(_get_labels_path(ctx, voice), labels)

    logger.info("{}/{} -> {}", voice, name, body.label)
    return {"name": name, "label": body.label}
