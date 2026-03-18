from __future__ import annotations

import asyncio
import subprocess
import threading
import wave
from io import BytesIO
from typing import Union

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names
from src.server.context import ServiceContext

router = APIRouter()


# ---------------------------------------------------------------------------
# Request 모델
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    voice: str
    text: str
    text_lang: str = "ko"
    emotion: str = "default"
    # --- 선택적 오버라이드 (voice.yaml 기본값 우선) ---
    ref_audio_path: str | None = None
    prompt_text: str | None = None
    prompt_lang: str | None = None
    # --- 합성 파라미터 ---
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    volume: float = 1.0


# ---------------------------------------------------------------------------
# 오디오 패킹 유틸
# ---------------------------------------------------------------------------

def _pack_ogg(
    io_buffer: BytesIO, data: np.ndarray, rate: int,
) -> BytesIO:
    def _write():
        with sf.SoundFile(
            io_buffer, mode="w", samplerate=rate,
            channels=1, format="ogg",
        ) as f:
            f.write(data)

    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        t = threading.Thread(target=_write)
        t.start()
        t.join()
    except (RuntimeError, ValueError) as e:
        logger.warning("ogg 패킹 스레드 스택 설정 실패: {}", e)
    return io_buffer


def _pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    io_buffer.write(data.tobytes())
    return io_buffer


def _pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def _pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    process = subprocess.Popen(
        [
            "ffmpeg", "-f", "s16le", "-ar", str(rate), "-ac", "1",
            "-i", "pipe:0", "-c:a", "aac", "-b:a", "192k",
            "-vn", "-f", "adts", "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def _pack_audio(
    io_buffer: BytesIO, data: np.ndarray,
    rate: int, media_type: str,
) -> BytesIO:
    packers = {"ogg": _pack_ogg, "aac": _pack_aac, "wav": _pack_wav}
    packer = packers.get(media_type, _pack_raw)
    io_buffer = packer(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def _wave_header_chunk(frame_input: bytes = b"", channels: int = 1,
                       sample_width: int = 2, sample_rate: int = 32000) -> bytes:
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


# ---------------------------------------------------------------------------
# 요청 검증
# ---------------------------------------------------------------------------

def _check_params(
    req: dict,
    supported_languages: list[str],
    cut_method_names: list[str],
) -> JSONResponse | None:
    text = req.get("text", "")
    text_lang = req.get("text_lang", "")
    ref_audio_path = req.get("ref_audio_path", "")
    media_type = req.get("media_type", "wav")
    prompt_lang = req.get("prompt_lang", "")
    text_split_method = req.get("text_split_method", "cut5")

    def _err(msg: str) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"message": msg},
        )

    if not ref_audio_path:
        return _err("ref_audio_path is required")
    if not text:
        return _err("text is required")
    if not text_lang:
        return _err("text_lang is required")
    if text_lang.lower() not in supported_languages:
        return _err(f"text_lang: {text_lang} is not supported")
    if not prompt_lang:
        return _err("prompt_lang is required")
    if prompt_lang.lower() not in supported_languages:
        return _err(f"prompt_lang: {prompt_lang} is not supported")
    if media_type not in ("wav", "raw", "ogg", "aac"):
        return _err(f"media_type: {media_type} is not supported")
    if text_split_method not in cut_method_names:
        return _err(
            f"text_split_method: {text_split_method}"
            " is not supported",
        )
    return None


# ---------------------------------------------------------------------------
# TTS 핸들러
# ---------------------------------------------------------------------------

def _resolve_streaming_mode(
    streaming_mode: bool | int,
) -> tuple[bool, bool, bool] | JSONResponse:
    """streaming_mode 값을 (streaming, return_fragment, fixed_length_chunk)로 변환."""
    return_fragment = False
    fixed_length_chunk = False

    if streaming_mode == 0 or streaming_mode is False:
        streaming_mode = False
    elif streaming_mode == 1 or streaming_mode is True:
        streaming_mode = False
        return_fragment = True
    elif streaming_mode == 2:
        streaming_mode = True
    elif streaming_mode == 3:
        streaming_mode = True
        fixed_length_chunk = True
    else:
        return JSONResponse(
            status_code=400,
            content={
                "message": "streaming_mode must be 0, 1, 2, 3"
                " (int) or true/false (bool)",
            },
        )

    return streaming_mode, return_fragment, fixed_length_chunk


def _apply_volume(audio_data: np.ndarray, volume: float) -> np.ndarray:
    """오디오 데이터에 볼륨 게인을 적용한다."""
    if volume == 1.0:
        return audio_data
    return np.clip(audio_data * volume, -1.0, 1.0)


async def _synthesize_stream(
    ctx: ServiceContext, voice_name: str, req: dict,
    media_type: str, volume: float = 1.0,
):
    """TTS 스트리밍 합성 (잠금 하에 실행)."""
    async with ctx.lock:
        await asyncio.to_thread(ctx.switch_voice, voice_name)
        gen = ctx.tts.synthesize(req)
        first = True
        while True:
            chunk = await asyncio.to_thread(lambda g=gen: next(g, None))
            if chunk is None:
                break
            sr, data = chunk
            if volume != 1.0:
                data = np.clip(data * volume, -1.0, 1.0)
            if first and media_type == "wav":
                yield _wave_header_chunk(sample_rate=sr)
                first = False
            yield _pack_audio(BytesIO(), data, sr, media_type).getvalue()


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


@router.post("/tts")
async def tts_post(request: Request, body: TTSRequest):
    """voice 기반 TTS 합성."""
    ctx = _get_context(request)

    if ctx.tts is None:
        raise HTTPException(
            503,
            detail="TTS 파이프라인이 초기화되지 않았습니다. pretrained 모델을 설치하세요.",
        )

    logger.info(
        "TTS 요청: voice={}, emotion={}, text='{}', lang={}, speed={}, volume={}, media={}",
        body.voice, body.emotion,
        body.text[:50] + "..." if len(body.text) > 50 else body.text,
        body.text_lang, body.speed_factor, body.volume, body.media_type,
    )

    # voice 조회
    profile = ctx.get_voice_profile(body.voice)
    if profile is None:
        return JSONResponse(
            status_code=404,
            content={"message": f"voice '{body.voice}' not found"},
        )

    # voice.yaml 기본값 + 요청 오버라이드
    emo_ref = profile.get_emotion(body.emotion)
    req = body.model_dump()
    req["ref_audio_path"] = body.ref_audio_path or emo_ref.ref_audio
    if body.prompt_text is not None:
        req["prompt_text"] = body.prompt_text
    else:
        req["prompt_text"] = emo_ref.ref_text
    req["prompt_lang"] = body.prompt_lang or profile.ref_lang
    volume = req.pop("volume", 1.0)

    media_type = req.get("media_type", "wav")

    # 파라미터 검증
    check_res = _check_params(req, ctx.tts_config.languages, get_method_names())
    if check_res is not None:
        return check_res

    result = _resolve_streaming_mode(req.get("streaming_mode", False))
    if isinstance(result, JSONResponse):
        return result
    streaming_mode, return_fragment, fixed_length_chunk = result
    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk
    is_streaming = streaming_mode or return_fragment

    try:
        if is_streaming:
            return StreamingResponse(
                _synthesize_stream(ctx, body.voice, req, media_type, volume),
                media_type=f"audio/{media_type}",
            )

        async with ctx.lock:
            def _synthesize():
                ctx.switch_voice(body.voice)
                gen = ctx.tts.synthesize(req)
                sr, audio_data = next(gen)
                audio_data = _apply_volume(audio_data, volume)
                return _pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()

            audio_bytes = await asyncio.to_thread(_synthesize)

        return Response(audio_bytes, media_type=f"audio/{media_type}")

    except Exception as e:
        logger.exception("TTS 합성 실패")
        return JSONResponse(
            status_code=400,
            content={"message": "tts failed", "exception": str(e)},
        )
