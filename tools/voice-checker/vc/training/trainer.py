"""CNN 모델 학습 루프."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.config.config import VoiceCheckerConfig as Config
from vc.data.dataset import AudioQualityDataset
from vc.data.transforms import AudioAugmentation, MelSpectrogramTransform
from vc.model.cnn import AudioQualityCNN


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_training(
    config: Config,
    data_dir: Path,
    labels_file: Path,
    output_dir: Path,
) -> None:
    """학습을 실행하고 최적 모델을 저장한다."""
    tc = config.training
    ac = config.augmentation

    torch.manual_seed(tc.seed)
    device = _detect_device()
    logger.info("디바이스: {}", device)

    # 데이터셋
    transform = MelSpectrogramTransform(config.audio)
    augmentation = AudioAugmentation(ac) if ac.enabled else None

    full_dataset = AudioQualityDataset(
        data_dir, labels_file, transform,
        augmentation=None,  # 검증셋은 증강 없이
        oversample_minority=1,
    )

    if len(full_dataset) == 0:
        logger.error("라벨된 데이터가 없습니다. 먼저 라벨링을 수행하세요.")
        return

    # train/val 분할 (stratified)
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    train_idx, val_idx = train_test_split(
        indices, test_size=tc.val_ratio,
        stratify=labels, random_state=tc.seed,
    )

    # 학습셋: 증강 + 오버샘플링 적용
    train_dataset = AudioQualityDataset(
        data_dir, labels_file, transform,
        augmentation=augmentation,
        oversample_minority=ac.minority_oversample if ac.enabled else 1,
    )

    # Subset으로 분할하되, 오버샘플링된 학습 데이터셋에서 원본 인덱스 기준으로 필터
    # 단순화: 전체를 학습+증강으로 사용하고 val은 원본 데이터셋에서 분할
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=tc.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=tc.batch_size,
        shuffle=False, num_workers=0,
    )

    # 모델
    model = AudioQualityCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("모델 파라미터: {:,}", param_count)

    # 클래스 가중치 (불균형 보정)
    good_count = sum(1 for _, l in full_dataset.samples if l == 0)
    bad_count = sum(1 for _, l in full_dataset.samples if l == 1)
    pos_weight = torch.tensor([good_count / max(bad_count, 1)], device=device)
    logger.info("클래스 비율: good {} / bad {} (pos_weight={:.2f})", good_count, bad_count, pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )

    # 학습 루프
    best_val_loss = float("inf")
    patience_counter = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, tc.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for mel, label in train_loader:
            mel = mel.to(device)
            label = label.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == label).sum().item()
            train_total += mel.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # -- Validation --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for mel, label in val_loader:
                mel = mel.to(device)
                label = label.float().unsqueeze(1).to(device)

                logits = model(mel)
                loss = criterion(logits, label)

                val_loss += loss.item() * mel.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += mel.size(0)

                all_preds.extend(preds.cpu().int().squeeze().tolist())
                all_labels.extend(label.cpu().int().squeeze().tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            "Epoch {}/{}: train_loss={:.4f} train_acc={:.3f} | val_loss={:.4f} val_acc={:.3f}",
            epoch, tc.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "n_mels": config.audio.n_mels,
                    "target_length": config.audio.target_length,
                    "sample_rate": config.audio.sample_rate,
                    "n_fft": config.audio.n_fft,
                    "hop_length": config.audio.hop_length,
                },
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_model_path)
            logger.info("  → best 모델 저장 (val_loss={:.4f})", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= tc.early_stop_patience:
                logger.info("Early stopping (patience={})", tc.early_stop_patience)
                break

    # 최종 리포트
    if all_labels:
        report = classification_report(
            all_labels, all_preds,
            target_names=["good", "bad"],
            zero_division=0,
        )
        logger.info("최종 분류 리포트:\n{}", report)

    logger.info("학습 완료 → {}", best_model_path)
