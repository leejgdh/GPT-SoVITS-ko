from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Dataclass 설정
# ---------------------------------------------------------------------------

@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 9880


@dataclass
class VCAudioConfig:
    sample_rate: int = 44100
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    target_length: int = 128


@dataclass
class VCTrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    val_ratio: float = 0.2
    early_stop_patience: int = 10
    seed: int = 42


@dataclass
class VCAugmentationConfig:
    enabled: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 5
    noise_std: float = 0.005
    minority_oversample: int = 5


@dataclass
class VCInferenceConfig:
    model_path: str = "data/voice-checker/models/best_model.pth"
    threshold: float = 0.5


@dataclass
class VoiceCheckerConfig:
    """Voice Checker 설정. conf.yaml에 voice_checker 섹션이 있으면 활성화."""
    audio: VCAudioConfig = field(default_factory=VCAudioConfig)
    training: VCTrainingConfig = field(default_factory=VCTrainingConfig)
    augmentation: VCAugmentationConfig = field(default_factory=VCAugmentationConfig)
    inference: VCInferenceConfig = field(default_factory=VCInferenceConfig)


@dataclass
class Config:
    service: ServiceConfig = field(default_factory=ServiceConfig)
    tts: dict = field(default_factory=dict)
    voices_dir: str = "data/voice"
    default_voice: str | None = None
    log_level: str = "INFO"
    voice_checker: VoiceCheckerConfig | None = None


def _find_latest(directory: str, pattern: str) -> str | None:
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
        gpt_path = _find_latest(os.path.join(step3, "02_gpt_weights"), "*.ckpt")
        if gpt_path:
            custom["t2s_weights_path"] = gpt_path
            logger.info("GPT 가중치 자동 탐색: {}", gpt_path)

    if "vits_weights_path" not in custom:
        sovits_path = _find_latest(os.path.join(step3, "04_sovits_weights"), "*.pth")
        if sovits_path:
            custom["vits_weights_path"] = sovits_path
            logger.info("SoVITS 가중치 자동 탐색: {}", sovits_path)


def load_config(path: Path) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    svc = data.get("service", {})
    tts = data.get("tts", {})

    if "custom" in tts:
        _resolve_voice_dir(tts["custom"])

    vc_data = data.get("voice_checker")
    vc_config = None
    if vc_data is not None:
        vc_config = VoiceCheckerConfig(
            audio=VCAudioConfig(**vc_data["audio"]) if "audio" in vc_data else VCAudioConfig(),
            training=VCTrainingConfig(**vc_data["training"]) if "training" in vc_data else VCTrainingConfig(),
            augmentation=VCAugmentationConfig(**vc_data["augmentation"]) if "augmentation" in vc_data else VCAugmentationConfig(),
            inference=VCInferenceConfig(**vc_data["inference"]) if "inference" in vc_data else VCInferenceConfig(),
        )

    return Config(
        service=ServiceConfig(
            host=svc.get("host", "0.0.0.0"),
            port=svc.get("port", 9880),
        ),
        tts=tts,
        voices_dir=data.get("voices_dir", "data/voice"),
        default_voice=data.get("default_voice"),
        log_level=data.get("log_level", "INFO"),
        voice_checker=vc_config,
    )


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


