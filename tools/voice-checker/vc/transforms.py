"""멜 스펙트로그램 변환 및 데이터 증강."""
from __future__ import annotations

import torch
import torchaudio

from src.config.config import VCAudioConfig as AudioConfig
from src.config.config import VCAugmentationConfig as AugmentationConfig


class MelSpectrogramTransform:
    """오디오 파형을 고정 크기 log-mel 스펙트로그램으로 변환한다.

    출력 shape: (1, n_mels, target_length)
    """

    def __init__(self, config: AudioConfig) -> None:
        self.sample_rate = config.sample_rate
        self.target_length = config.target_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # 리샘플링 (필요 시)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # 스테레오 → 모노
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 멜 스펙트로그램 → log scale
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)

        # 시간축 패딩/트림 → (1, n_mels, target_length)
        _, n_mels, time_len = mel.shape
        if time_len < self.target_length:
            pad = self.target_length - time_len
            mel = torch.nn.functional.pad(mel, (0, pad))
        elif time_len > self.target_length:
            mel = mel[:, :, :self.target_length]

        # 정규화 (mean=0, std=1)
        mean = mel.mean()
        std = mel.std()
        if std > 0:
            mel = (mel - mean) / std

        return mel


class AudioAugmentation:
    """학습 시 적용할 데이터 증강 (SpecAugment + 가우시안 노이즈)."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.time_masking = torchaudio.transforms.TimeMasking(config.time_mask_param)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(config.freq_mask_param)
        self.noise_std = config.noise_std

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = self.time_masking(mel)
        mel = self.freq_masking(mel)
        if self.noise_std > 0:
            mel = mel + torch.randn_like(mel) * self.noise_std
        return mel
