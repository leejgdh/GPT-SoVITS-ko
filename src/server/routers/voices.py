from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.server.context import ServiceContext

router = APIRouter()


def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


@router.get("/voices")
async def list_voices(request: Request):
    """등록된 voice 목록을 반환한다."""
    ctx = _get_context(request)
    voices = [
        {"name": p.name, "version": p.version, "ref_lang": p.ref_lang, "available": p.available}
        for p in ctx.voices.values()
    ]
    return JSONResponse(content={
        "voices": voices,
        "current_voice": ctx.current_voice,
    })


@router.get("/voices/{name}")
async def get_voice(request: Request, name: str):
    """특정 voice의 상세 정보를 반환한다."""
    ctx = _get_context(request)
    profile = ctx.get_voice_profile(name)
    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"message": f"voice '{name}' not found"},
        )
    return JSONResponse(content={
        "name": profile.name,
        "version": profile.version,
        "ref_lang": profile.ref_lang,
        "emotions": profile.emotion_names,
    })
