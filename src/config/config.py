"""tts-service 설정 모델 + 디바이스 감지 helper.

pydantic v2 BaseModel 기반 — VCAugmentationConfig.enabled 등 bool 필드의 env
치환 함정 (``bool("false") == True``) 을 BoolValidator 가 자동 해결.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import torch
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.exceptions import ConfigError

# ---------------------------------------------------------------------------
# BaseModel 설정
# ---------------------------------------------------------------------------


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9880


class VCAudioConfig(BaseModel):
    sample_rate: int = 44100
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    target_length: int = 128


class VCTrainingConfig(BaseModel):
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    early_stop_patience: int = Field(default=10, ge=0)
    seed: int = 42


class VCAugmentationConfig(BaseModel):
    enabled: bool = True
    time_mask_param: int = Field(default=10, ge=0)
    freq_mask_param: int = Field(default=5, ge=0)
    noise_std: float = Field(default=0.005, ge=0.0)
    minority_oversample: int = Field(default=5, ge=0)


class VCInferenceConfig(BaseModel):
    model_path: str = "data/voice-checker/models/best_model.pth"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class VoiceCheckerConfig(BaseModel):
    """Voice Checker 설정. config.yaml에 voice_checker 섹션이 있으면 활성화."""

    audio: VCAudioConfig = Field(default_factory=VCAudioConfig)
    training: VCTrainingConfig = Field(default_factory=VCTrainingConfig)
    augmentation: VCAugmentationConfig = Field(default_factory=VCAugmentationConfig)
    inference: VCInferenceConfig = Field(default_factory=VCInferenceConfig)


class Config(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    # tts 는 GPT-SoVITS 원본의 자유 dict — version / device / custom 등 양식 다양.
    # BaseModel 로 묶기엔 키 집합이 너무 가변적이라 raw dict 유지.
    tts: dict = Field(default_factory=dict)
    voices_dir: str = "data/voice"
    default_voice: str | None = None
    log_level: str = "INFO"
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
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        cfg = Config.model_validate(data)
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
    if mem_gb < 4 or sm_version < 5.3:
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
