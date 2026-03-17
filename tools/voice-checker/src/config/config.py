"""설정 관리.

conf.yaml을 읽어 @dataclass Config 객체로 변환한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    target_length: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    val_ratio: float = 0.2
    early_stop_patience: int = 10
    seed: int = 42


@dataclass
class AugmentationConfig:
    enabled: bool = True
    time_mask_param: int = 10
    freq_mask_param: int = 5
    noise_std: float = 0.005
    minority_oversample: int = 5


@dataclass
class InferenceConfig:
    model_path: str = "models/best_model.pth"
    threshold: float = 0.5


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 9890


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    log_level: str = "INFO"


def load_config(path: Path) -> Config:
    """YAML 설정 파일을 로드하여 Config 객체를 반환한다."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        audio=AudioConfig(**raw["audio"]) if "audio" in raw else AudioConfig(),
        training=TrainingConfig(**raw["training"]) if "training" in raw else TrainingConfig(),
        augmentation=AugmentationConfig(**raw["augmentation"]) if "augmentation" in raw else AugmentationConfig(),
        inference=InferenceConfig(**raw["inference"]) if "inference" in raw else InferenceConfig(),
        service=ServiceConfig(**raw["service"]) if "service" in raw else ServiceConfig(),
        log_level=raw.get("log_level", "INFO"),
    )
