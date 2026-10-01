"""tts-service 설정 모델 + 디바이스 감지 helper.

YAML 값의 ``${VAR}`` 를 환경변수 값으로 치환한다. 기본값 양식(``${VAR:-값}``)은
받지 않고, 환경변수가 없으면 ``ConfigError`` 로 멈춘다 — 값이 사는 곳은 프로젝트
루트 `.env` 하나다 (컨테이너 실행은 compose 의 environment 가 주입).

설정 모델에도 기본값을 두지 않는다. config.yaml 이 값을 다 적고, 빠지면 검증이
실패한다 — 같은 값이 코드와 yaml 두 곳에 적히지 않게 한다.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.exceptions import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _replace(m: re.Match[str]) -> str:
    var, default = m.group(1), m.group(2)
    if default is not None:
        raise ConfigError(
            f"{var} 에 기본값이 붙어 있습니다 — 기본값은 코드에 두지 않고 "
            f"`.env` 에 값을 넣습니다 (config.yaml 에서 `:-` 를 지워야 합니다)",
        )
    env_val = os.environ.get(var)
    if env_val is None:
        raise ConfigError(
            f"환경변수 {var} 가 없습니다 — 프로젝트 루트 `.env` 에 값을 넣어야 합니다 "
            f"(컨테이너 실행은 compose 의 environment 가 주입)",
        )
    return env_val


def _expand_env(value: Any) -> Any:
    """문자열/dict/list 안의 ``${VAR}`` 를 환경변수 값으로 치환한다.

    기본값 양식(``${VAR:-값}``)은 받지 않는다 — 값이 사는 곳은 `.env` 하나다.
    """
    if isinstance(value, str):
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# BaseModel 설정 — 기본값을 두지 않는다. config.yaml 이 값을 다 적는다.
# ---------------------------------------------------------------------------


class ServiceConfig(BaseModel):
    host: str
    port: int


class VCAudioConfig(BaseModel):
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    target_length: int


class VCTrainingConfig(BaseModel):
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    val_ratio: float = Field(ge=0.0, lt=1.0)
    early_stop_patience: int = Field(ge=0)
    seed: int


class VCAugmentationConfig(BaseModel):
    enabled: bool
    time_mask_param: int = Field(ge=0)
    freq_mask_param: int = Field(ge=0)
    noise_std: float = Field(ge=0.0)
    minority_oversample: int = Field(ge=0)


class VCInferenceConfig(BaseModel):
    model_path: str
    threshold: float = Field(ge=0.0, le=1.0)


class VoiceCheckerConfig(BaseModel):
    """Voice Checker 설정. config.yaml에 voice_checker 섹션이 있으면 활성화."""

    audio: VCAudioConfig
    training: VCTrainingConfig
    augmentation: VCAugmentationConfig
    inference: VCInferenceConfig


class Config(BaseModel):
    service: ServiceConfig
    # tts 는 GPT-SoVITS 원본의 자유 dict — version / device / custom 등 양식 다양.
    # BaseModel 로 묶기엔 키 집합이 너무 가변적이라 raw dict 유지.
    # 섹션이 없으면 device / is_half 를 실행 환경에서 감지한다.
    tts: dict = Field(default_factory=dict)
    voices_dir: str
    # 빈 값이면 기동 시 voice 를 미리 올리지 않는다.
    default_voice: str | None
    log_level: str
    # 섹션이 없으면 Voice Checker 를 쓰지 않는다.
    voice_checker: VoiceCheckerConfig | None = None


def find_latest_weight(directory: str, pattern: str) -> str | None:
    """디렉토리에서 가장 최근 수정된 파일을 찾는다."""
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None


def _resolve_voice_dir(custom: dict) -> None:
    """voice_dir 키로부터 가중치 경로를 자동 탐색하여 설정한다."""
    voice_dir = custom.pop("voice_dir", None)
    if voice_dir is None:
        return

    version = custom.get("version", "v2Pro")
    step3 = os.path.join(voice_dir, "step3", version)

    if "t2s_weights_path" not in custom:
        gpt_path = find_latest_weight(os.path.join(step3, "02_gpt_weights"), "*.ckpt")
        if gpt_path:
            custom["t2s_weights_path"] = gpt_path
            logger.info("GPT 가중치 자동 탐색: {}", gpt_path)

    if "vits_weights_path" not in custom:
        sovits_path = find_latest_weight(os.path.join(step3, "04_sovits_weights"), "*.pth")
        if sovits_path:
            custom["vits_weights_path"] = sovits_path
            logger.info("SoVITS 가중치 자동 탐색: {}", sovits_path)


def load_config(path: Path) -> Config:
    """yaml 파일 한 번 ``Config.model_validate`` 로 전체 변환.

    tts dict 는 GPT-SoVITS 원본의 free-form 양식이라 BaseModel 로 묶지 않고
    그대로 dict 유지 — 그 안의 키 (custom / version / weights 등) 는 free-form
    영역이므로 dict 접근 양식 OK.
    """
    if not path.exists():
        raise ConfigError(f"설정 파일 없음: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        cfg = Config.model_validate(_expand_env(data))
    except ValidationError as e:
        raise ConfigError(f"설정 검증 실패: {e}") from e

    # tts.custom.voice_dir 자동 탐색 — tts 는 free-form 양식이라 dict 접근 영역.
    custom = cfg.tts.get("custom")
    if isinstance(custom, dict):
        _resolve_voice_dir(custom)

    return cfg


# ---------------------------------------------------------------------------
# Pretrained 모델 경로 상수
# ---------------------------------------------------------------------------

pretrained_sovits_name: dict[str, str] = {
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
}

pretrained_gpt_name: dict[str, str] = {
    "v2": (
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained"
        "/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    ),
    "v3": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v4": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2Pro": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}

# ---------------------------------------------------------------------------
# GPU / Device 감지
# ---------------------------------------------------------------------------


def get_device_dtype_sm(idx: int) -> tuple[torch.device, torch.dtype, float, float]:
    cpu = torch.device("cpu")
    cuda = torch.device(f"cuda:{idx}")
    if not torch.cuda.is_available():
        return cpu, torch.float32, 0.0, 0.0
    capability = torch.cuda.get_device_capability(idx)
    name = torch.cuda.get_device_name(idx)
    mem_bytes = torch.cuda.get_device_properties(idx).total_memory
    mem_gb = mem_bytes / (1024**3) + 0.4
    major, minor = capability
    sm_version = major + minor / 10.0
    is_16_series = bool(re.search(r"16\d{2}", name)) and sm_version == 7.5
    # sm_version 가드만 유지 — Maxwell 이전 (sm < 5.3) 은 float 연산 자체가
    # 불안정. VRAM 사이즈 가드는 제거 — 작은 GPU 도 시도 가능 (모델 로드 시점에
    # OOM 으로 떨어지더라도 정책적 차단 X).
    if sm_version < 5.3:
        return cpu, torch.float32, 0.0, 0.0
    if sm_version == 6.1 or is_16_series:
        return cuda, torch.float32, sm_version, mem_gb
    if sm_version > 6.1:
        return cuda, torch.float16, sm_version, mem_gb
    return cpu, torch.float32, 0.0, 0.0


def detect_device() -> torch.device:
    """최적의 추론 디바이스를 반환한다."""
    gpu_count = torch.cuda.device_count()
    results = [get_device_dtype_sm(i) for i in range(max(gpu_count, 1))]
    best = max(results, key=lambda x: (x[2], x[3]))
    return best[0]


def detect_half() -> bool:
    """half-precision(float16) 사용 가능 여부를 반환한다."""
    gpu_count = torch.cuda.device_count()
    results = [get_device_dtype_sm(i) for i in range(max(gpu_count, 1))]
    return any(dtype == torch.float16 for _, dtype, _, _ in results)
