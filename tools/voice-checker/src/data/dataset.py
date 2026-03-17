"""오디오 품질 분류 데이터셋."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
from loguru import logger
from torch.utils.data import Dataset

from src.data.transforms import AudioAugmentation, MelSpectrogramTransform


class AudioQualityDataset(Dataset):
    """labels.json 기반 오디오 품질 이진 분류 데이터셋.

    라벨: 0 = good (정상), 1 = bad (비정상)
    unlabeled은 제외된다.
    """

    def __init__(
        self,
        data_dir: Path,
        labels_file: Path,
        transform: MelSpectrogramTransform,
        augmentation: AudioAugmentation | None = None,
        oversample_minority: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = augmentation

        # labels.json 파싱 — good/bad만 사용
        with open(labels_file, encoding="utf-8") as f:
            entries = json.load(f).get("files", [])

        self.samples: list[tuple[str, int]] = []
        good_indices: list[int] = []
        bad_indices: list[int] = []

        for entry in entries:
            label_str = entry.get("label", "unlabeled")
            if label_str == "unlabeled":
                continue
            audio_path = str(data_dir / entry["name"])
            label = 0 if label_str == "good" else 1
            idx = len(self.samples)
            self.samples.append((audio_path, label))
            if label == 0:
                good_indices.append(idx)
            else:
                bad_indices.append(idx)

        # 소수 클래스 오버샘플링
        if oversample_minority > 1 and bad_indices:
            minority = bad_indices if len(bad_indices) < len(good_indices) else good_indices
            extra = minority * (oversample_minority - 1)
            self.samples.extend(self.samples[i] for i in extra)

        logger.info(
            "데이터셋 구성: good {} / bad {} / 오버샘플 후 전체 {}",
            len(good_indices), len(bad_indices), len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]

        waveform, sr = torchaudio.load(audio_path)
        mel = self.transform(waveform, sr)

        if self.augmentation is not None:
            mel = self.augmentation(mel)

        return mel, label
