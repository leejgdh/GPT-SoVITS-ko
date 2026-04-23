"""tts-service Prometheus 메트릭 정의.

Prometheus text exposition format 으로 ``/metrics`` 엔드포인트에 노출된다.
수집 전용 — 라우터/컨텍스트가 호출해 카운트·관측만 수행한다.

메트릭 이름 규칙: ``tts_service_<subsystem>_<name>_<unit>``

라벨 카디널리티 주의:
- ``voice`` 는 등록된 음성 프로필 수 (일반적으로 5~20개) 라 라벨로 안전.
- ``path`` 는 라우트 템플릿 (``/voices/{name}/labels`` 등) 으로 통일.
- text 내용이나 timestamp 같이 기수 큰 값은 라벨로 쓰지 않는다.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_NS = "tts_service"

# ── 공통: 프로세스/HTTP ────────────────────────────────────────────────────

process_up = Gauge(
    f"{_NS}_up",
    "1 if the process is running and the lifespan has started.",
)

http_requests_total = Counter(
    f"{_NS}_http_requests_total",
    "Total HTTP requests handled, labelled by route template and result.",
    ["method", "path", "result"],
)

http_duration_seconds = Histogram(
    f"{_NS}_http_duration_seconds",
    "HTTP request duration end-to-end (middleware).",
    ["method", "path", "result"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ── 특화: TTS 합성 ────────────────────────────────────────────────────────

synthesis_duration_seconds = Histogram(
    f"{_NS}_synthesis_duration_seconds",
    "End-to-end /tts synthesis duration (voice switch + inference + encode).",
    ["voice", "result"],   # result=ok|error
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0),
)

synthesis_chars = Histogram(
    f"{_NS}_synthesis_chars",
    "Input text length (chars) per /tts synthesis. Combined with duration "
    "it yields chars-per-second throughput.",
    ["voice"],
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048),
)

synthesis_bytes = Histogram(
    f"{_NS}_synthesis_bytes",
    "Output audio payload size (bytes) per /tts synthesis.",
    ["voice"],
    buckets=(
        32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
        1 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024,
        8 * 1024 * 1024,
    ),
)

# ── 특화: voice 전환 (VRAM 스왑 체감) ─────────────────────────────────────

voice_switch_total = Counter(
    f"{_NS}_voice_switch_total",
    "Voice profile switches performed (load weights into VRAM).",
    ["voice"],
)

active_voice = Gauge(
    f"{_NS}_active_voice",
    "1 if the given voice is currently loaded in memory, else 0.",
    ["voice"],
)


__all__ = [
    "process_up",
    "http_requests_total",
    "http_duration_seconds",
    "synthesis_duration_seconds",
    "synthesis_chars",
    "synthesis_bytes",
    "voice_switch_total",
    "active_voice",
]
