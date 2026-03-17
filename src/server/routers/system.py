from __future__ import annotations

import os
import signal
import sys

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_REVIEW_HTML = os.path.join(_PROJECT_ROOT, "tools", "label-review.html")


@router.get("/health")
async def health():
    """헬스체크 엔드포인트."""
    return {"status": "ok"}


@router.get("/review", response_class=HTMLResponse)
async def review_page():
    """라벨 검수 UI를 서빙한다."""
    with open(_REVIEW_HTML, encoding="utf-8") as f:
        html = f.read()
    # 외부 접속 시 서버 주소를 자동으로 현재 호스트로 설정
    html = html.replace(
        'value="http://localhost:9880"',
        'value=""',
    )
    html = html.replace(
        "document.addEventListener('DOMContentLoaded', connect);",
        "document.addEventListener('DOMContentLoaded', () => {"
        " $('#server-url').value = window.location.origin;"
        " connect();"
        " });",
    )
    return HTMLResponse(content=html)


@router.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
