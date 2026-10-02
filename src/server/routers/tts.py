from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import time
import wave
from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names
from src.metrics import (
    synthesis_bytes,
    synthesis_chars,
    synthesis_duration_seconds,
)
from src.server.context import ServiceContext
from src.server.paths import is_within

router = APIRouter()


# ---------------------------------------------------------------------------
# Request 모델
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """OpenAI ``POST /v1/audio/speech`` 요청. 표준 필드 아래는 확장 필드다.

    확장 필드는 OpenAI SDK 의 ``extra_body`` 로 보내면 최상위 키로 합쳐져 도착한다.
    """

    # --- OpenAI 표준 ---
    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: str = "audio"
    # --- 확장: 화자·언어 ---
    emotion: str = "default"
    text_lang: str = "ko"
    # --- 확장: voice.yaml 기본값 오버라이드 ---
    ref_audio_path: str | None = None
    prompt_text: str | None = None
    prompt_lang: str | None = None
    # --- 확장: 합성 파라미터 ---
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    fragment_interval: float = 0.3
    seed: int = -1
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    volume: float = 1.0


# OpenAI response_format → 내부 packer 키 / Content-Type
_RESPONSE_FORMATS: dict[str, tuple[str, str]] = {
    "mp3": ("mp3", "audio/mpeg"),
    "wav": ("wav", "audio/wav"),
    "aac": ("aac", "audio/aac"),
    "pcm": ("raw", "audio/pcm"),
}
_STREAM_FORMATS = ("audio", "sse")


def openai_error(
    status_code: int, message: str, *,
    param: str | None = None, code: str | None = None,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    """OpenAI 오류 본문 ``{"error": {message, type, param, code}}`` 응답."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {
            "message": message, "type": error_type, "param": param, "code": code,
        }},
    )


# ---------------------------------------------------------------------------
# 오디오 패킹 유틸
# ---------------------------------------------------------------------------

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
    out, err = process.communicate(input=data.tobytes())
    if process.returncode != 0:
        msg = f"ffmpeg aac 인코딩 실패 (code={process.returncode}): {err.decode(errors='replace')[-300:]}"
        raise RuntimeError(msg)
    io_buffer.write(out)
    return io_buffer


def _pack_mp3(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    process = subprocess.Popen(
        [
            "ffmpeg", "-f", "s16le", "-ar", str(rate), "-ac", "1",
            "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "192k",
            "-vn", "-f", "mp3", "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate(input=data.tobytes())
    if process.returncode != 0:
        msg = f"ffmpeg mp3 인코딩 실패 (code={process.returncode}): {err.decode(errors='replace')[-300:]}"
        raise RuntimeError(msg)
    io_buffer.write(out)
    return io_buffer


def _pack_audio(
    io_buffer: BytesIO, data: np.ndarray,
    rate: int, media_type: str,
) -> BytesIO:
    packers = {"aac": _pack_aac, "mp3": _pack_mp3, "wav": _pack_wav}
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
    body: SpeechRequest,
    req: dict,
    supported_languages: list[str],
    cut_method_names: list[str],
) -> JSONResponse | None:
    """검증 실패 시 OpenAI 오류 응답을, 통과하면 None 을 반환한다."""
    if not body.input:
        return openai_error(400, "input is required", param="input")
    if body.response_format not in _RESPONSE_FORMATS:
        return openai_error(
            400,
            f"response_format: {body.response_format} is not supported"
            f" (supported: {', '.join(_RESPONSE_FORMATS)})",
            param="response_format", code="unsupported_format",
        )
    if body.stream_format not in _STREAM_FORMATS:
        return openai_error(
            400,
            f"stream_format: {body.stream_format} is not supported"
            f" (supported: {', '.join(_STREAM_FORMATS)})",
            param="stream_format", code="unsupported_format",
        )
    if not req.get("ref_audio_path"):
        return openai_error(400, "ref_audio_path is required", param="ref_audio_path")
    if body.text_lang.lower() not in supported_languages:
        return openai_error(
            400, f"text_lang: {body.text_lang} is not supported", param="text_lang",
        )
    if not req.get("prompt_lang"):
        return openai_error(400, "prompt_lang is required", param="prompt_lang")
    if req["prompt_lang"].lower() not in supported_languages:
        return openai_error(
            400, f"prompt_lang: {req['prompt_lang']} is not supported", param="prompt_lang",
        )
    if body.text_split_method not in cut_method_names:
        return openai_error(
            400, f"text_split_method: {body.text_split_method} is not supported",
            param="text_split_method",
        )
    return None


# ---------------------------------------------------------------------------
# TTS 핸들러
# ---------------------------------------------------------------------------

def _apply_volume(audio_data: np.ndarray, volume: float) -> np.ndarray:
    """오디오 데이터에 볼륨 게인을 적용한다."""
    if volume == 1.0:
        return audio_data
    return np.clip(audio_data * volume, -1.0, 1.0)


async def _synthesize_chunks(
    ctx: ServiceContext, voice_name: str, req: dict,
    media_type: str, volume: float = 1.0,
):
    """합성 fragment 를 인코딩해 순서대로 내보낸다 (잠금 하에 실행).

    합성 generator 의 next() 와 chunk 인코딩 (_pack_audio) 모두 to_thread 로
    감싸 event loop 블로킹을 회피한다. aac 는 인코더 호출이 무거워 특히 중요.
    wav 는 첫 chunk 앞에 헤더 chunk 를 하나 내보낸다.
    """
    async with ctx.lock:
        await asyncio.to_thread(ctx.switch_voice, voice_name)
        gen = ctx.tts.synthesize(req)
        first = True
        while True:
            chunk = await asyncio.to_thread(lambda g=gen: next(g, None))
            if chunk is None:
                break
            sr, data = chunk
            data = _apply_volume(data, volume)
            if first and media_type == "wav":
                yield _wave_header_chunk(sample_rate=sr)
                first = False
            yield await asyncio.to_thread(
                lambda d=data, r=sr: _pack_audio(BytesIO(), d, r, media_type).getvalue()
            )


async def _sse_events(chunks, voice_name: str, text: str):
    """인코딩된 chunk 를 OpenAI ``speech.audio.delta`` / ``speech.audio.done`` 이벤트로 감싼다.

    합성 소요 시간은 마지막 chunk 까지 측정해 기록한다.
    """
    t0 = time.monotonic()
    result = "error"
    total = 0
    try:
        async for chunk in chunks:
            total += len(chunk)
            event = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode()}
            yield f"data: {json.dumps(event)}\n\n"
        result = "ok"
    finally:
        synthesis_duration_seconds.labels(voice=voice_name, result=result).observe(
            time.monotonic() - t0,
        )
    synthesis_bytes.labels(voice=voice_name).observe(total)
    done = {"type": "speech.audio.done", "usage": {
        "input_tokens": len(text), "output_tokens": total, "total_tokens": len(text) + total,
    }}
    yield f"data: {json.dumps(done)}\n\n"


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

def _get_context(request: Request) -> ServiceContext:
    return request.app.state.context


@router.post("/v1/audio/speech")
async def create_speech(request: Request, body: SpeechRequest):
    """OpenAI 호환 음성 합성. ``voice`` 는 Voice Profile 이름이다."""
    ctx = _get_context(request)

    if ctx.tts is None:
        return openai_error(
            503,
            "TTS 파이프라인이 초기화되지 않았습니다. pretrained 모델을 설치하세요.",
            error_type="server_error", code="service_unavailable",
        )

    logger.info(
        "TTS 요청: voice={}, emotion={}, input='{}', lang={}, speed={}, volume={}, format={}, stream={}",
        body.voice, body.emotion,
        body.input[:50] + "..." if len(body.input) > 50 else body.input,
        body.text_lang, body.speed, body.volume, body.response_format, body.stream_format,
    )

    profile = ctx.get_voice_profile(body.voice)
    if profile is None:
        return openai_error(
            404, f"voice '{body.voice}' not found", param="voice", code="voice_not_found",
        )
    if not profile.available:
        return openai_error(
            503, f"voice '{body.voice}'는 아직 학습이 완료되지 않았습니다 (available: false)",
            error_type="server_error", code="voice_unavailable",
        )

    # voice.yaml 기본값 + 요청 오버라이드
    if body.ref_audio_path and not is_within(ctx.config.voices_dir, body.ref_audio_path):
        return openai_error(
            400, "ref_audio_path 는 voices_dir 하위여야 합니다", param="ref_audio_path",
        )
    emo_ref = profile.get_emotion(body.emotion)
    req = body.model_dump(exclude={"model", "input", "response_format", "speed", "stream_format"})
    req["text"] = body.input
    req["speed_factor"] = body.speed
    req["ref_audio_path"] = body.ref_audio_path or emo_ref.ref_audio
    req["prompt_text"] = body.prompt_text if body.prompt_text is not None else emo_ref.ref_text
    req["prompt_lang"] = body.prompt_lang or profile.ref_lang
    volume = req.pop("volume")

    check_res = _check_params(body, req, ctx.tts_config.languages, get_method_names())
    if check_res is not None:
        return check_res

    media_type, content_type = _RESPONSE_FORMATS[body.response_format]
    is_sse = body.stream_format == "sse"
    req["streaming_mode"] = False
    req["return_fragment"] = is_sse
    req["fixed_length_chunk"] = False

    synthesis_chars.labels(voice=body.voice).observe(len(body.input))

    t0 = time.monotonic()
    try:
        if is_sse:
            chunks = _synthesize_chunks(ctx, body.voice, req, media_type, volume)
            return StreamingResponse(
                _sse_events(chunks, body.voice, body.input),
                media_type="text/event-stream",
            )

        async with ctx.lock:
            def _synthesize():
                ctx.switch_voice(body.voice)
                gen = ctx.tts.synthesize(req)
                sr, audio_data = next(gen)
                audio_data = _apply_volume(audio_data, volume)
                return _pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()

            audio_bytes = await asyncio.to_thread(_synthesize)

        elapsed = time.monotonic() - t0
        synthesis_duration_seconds.labels(voice=body.voice, result="ok").observe(elapsed)
        synthesis_bytes.labels(voice=body.voice).observe(len(audio_bytes))
        return Response(audio_bytes, media_type=content_type)

    except Exception:
        synthesis_duration_seconds.labels(
            voice=body.voice, result="error"
        ).observe(time.monotonic() - t0)
        logger.exception("TTS 합성 실패")
        return openai_error(
            500, "tts failed", error_type="server_error", code="synthesis_failed",
        )
