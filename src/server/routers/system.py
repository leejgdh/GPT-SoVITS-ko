from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_REVIEW_HTML = os.path.join(_PROJECT_ROOT, "tools", "label-review.html")
_DEMO_HTML = os.path.join(_PROJECT_ROOT, "tools", "tts-demo.html")


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


@router.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """TTS 데모 UI 를 서빙한다. 텍스트 입력 → voice/emotion 선택 → inline 재생."""
    with open(_DEMO_HTML, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# /control (restart/exit) 엔드포인트는 인증 게이트가 없어 외부 노출 시 DoS/프로세스
# 탈취 위험이 있어 제거했다. 재시작/종료는 docker compose 나 systemd 로 관리한다.
