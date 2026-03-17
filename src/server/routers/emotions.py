from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.config.voice import EmotionRef, save_voice_yaml
from src.server.context import ServiceContext
from src.server.routers.labels import get_label_path, read_labels

router = APIRouter()

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


# ---------------------------------------------------------------------------
# Request 모델
# ---------------------------------------------------------------------------

class EmotionMapRequest(BaseModel):
    label_index: int  # 03_vocal.list의 라벨 인덱스


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


@router.get("/voices/{name}/emotions")
async def list_emotions(request: Request, name: str):
    """voice에 등록된 감정 매핑 목록을 반환한다."""
    ctx = _get_context(request)
    profile = ctx.get_voice_profile(name)
    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"message": f"voice '{name}' not found"},
        )

    emotions = {
        emo_name: {
            "ref_audio": os.path.basename(emo.ref_audio),
            "ref_text": emo.ref_text,
        }
        for emo_name, emo in profile.emotions.items()
    }
    return JSONResponse(content={
        "voice": name,
        "emotions": emotions,
    })


@router.put("/voices/{name}/emotions/{emotion}")
async def set_emotion(
    request: Request, name: str, emotion: str,
    body: EmotionMapRequest,
):
    """라벨 인덱스를 감정에 매핑하여 voice.yaml에 저장한다."""
    ctx = _get_context(request)
    profile = ctx.get_voice_profile(name)
    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"message": f"voice '{name}' not found"},
        )

    # 라벨에서 audio/text 가져오기
    label_path = get_label_path(ctx.config.voices_dir, name)
    if not os.path.isfile(label_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"라벨 파일을 찾을 수 없습니다: {name}"},
        )

    labels = read_labels(label_path)
    if body.label_index < 0 or body.label_index >= len(labels):
        return JSONResponse(
            status_code=400,
            content={
                "message": f"인덱스 범위 초과: {body.label_index}"
                f" (총 {len(labels)}개)",
            },
        )

    label = labels[body.label_index]
    voice_dir = os.path.join(ctx.config.voices_dir, name)

    # 프로필에 감정 추가/갱신
    ref_audio_path = label["path"]
    if not os.path.isabs(ref_audio_path):
        ref_audio_path = os.path.join(_PROJECT_ROOT, ref_audio_path)

    profile.emotions[emotion] = EmotionRef(
        ref_audio=ref_audio_path,
        ref_text=label["text"],
    )

    # voice.yaml 저장 (상대경로로 변환)
    emotions_data = {
        emo_name: {
            "ref_audio": os.path.relpath(emo.ref_audio, voice_dir),
            "ref_text": emo.ref_text,
        }
        for emo_name, emo in profile.emotions.items()
    }
    save_voice_yaml(
        voice_dir,
        name=profile.name,
        version=profile.version,
        ref_audio=profile.ref_audio,
        ref_text=profile.ref_text,
        ref_lang=profile.ref_lang,
        gpt_weights=profile.gpt_weights,
        sovits_weights=profile.sovits_weights,
        emotions=emotions_data,
    )

    logger.info(
        "감정 매핑: voice={}, emotion={}, audio={}",
        name, emotion, os.path.basename(ref_audio_path),
    )

    return JSONResponse(content={
        "message": "감정 매핑 완료",
        "emotion": emotion,
        "ref_audio": os.path.basename(ref_audio_path),
        "ref_text": label["text"],
    })


@router.delete("/voices/{name}/emotions/{emotion}")
async def delete_emotion(request: Request, name: str, emotion: str):
    """감정 매핑을 삭제한다."""
    ctx = _get_context(request)
    profile = ctx.get_voice_profile(name)
    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"message": f"voice '{name}' not found"},
        )

    if emotion not in profile.emotions:
        return JSONResponse(
            status_code=404,
            content={"message": f"감정 '{emotion}'이 등록되어 있지 않습니다"},
        )

    if emotion == "default" and len(profile.emotions) == 1:
        return JSONResponse(
            status_code=400,
            content={"message": "default 감정은 마지막 하나일 때 삭제할 수 없습니다"},
        )

    del profile.emotions[emotion]
    voice_dir = os.path.join(ctx.config.voices_dir, name)

    emotions_data = {
        emo_name: {
            "ref_audio": os.path.relpath(emo.ref_audio, voice_dir),
            "ref_text": emo.ref_text,
        }
        for emo_name, emo in profile.emotions.items()
    }
    save_voice_yaml(
        voice_dir,
        name=profile.name,
        version=profile.version,
        ref_audio=profile.ref_audio,
        ref_text=profile.ref_text,
        ref_lang=profile.ref_lang,
        gpt_weights=profile.gpt_weights,
        sovits_weights=profile.sovits_weights,
        emotions=emotions_data,
    )

    logger.info("감정 삭제: voice={}, emotion={}", name, emotion)

    return JSONResponse(content={
        "message": "감정 삭제 완료",
        "emotion": emotion,
        "remaining": profile.emotion_names,
    })
