"""경량 2D CNN 오디오 품질 이진 분류기."""
from __future__ import annotations

import torch
import torch.nn as nn


class AudioQualityCNN(nn.Module):
    """멜 스펙트로그램 기반 이진 분류 CNN.

    입력: (B, 1, n_mels, target_length) — 기본 (B, 1, 64, 128)
    출력: (B, 1) — logit (sigmoid 전)

    아키텍처:
      Conv2d(1→16) → BN → ReLU → MaxPool(2)
      Conv2d(16→32) → BN → ReLU → MaxPool(2)
      Conv2d(32→64) → BN → ReLU → AdaptiveAvgPool(4,4)
      Flatten → Dropout(0.5) → Linear(1024→64) → ReLU → Linear(64→1)
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
