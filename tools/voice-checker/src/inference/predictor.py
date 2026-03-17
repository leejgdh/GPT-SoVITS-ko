"""오디오 품질 추론."""
from __future__ import annotations

import torch
import torchaudio
from loguru import logger

from src.config.config import AudioConfig, Config
from src.data.transforms import MelSpectrogramTransform
from src.model.cnn import AudioQualityCNN


class VoiceQualityPredictor:
    """학습된 CNN 모델로 오디오 품질을 예측한다.

    사용법::

        predictor = VoiceQualityPredictor("models/best_model.pth", config)
        is_good, confidence = predictor.predict("audio.wav")
    """

    def __init__(self, model_path: str, config: Config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # 체크포인트에 저장된 오디오 설정 우선 사용
        saved_cfg = checkpoint.get("config", {})
        audio_config = AudioConfig(
            sample_rate=saved_cfg.get("sample_rate", config.audio.sample_rate),
            n_mels=saved_cfg.get("n_mels", config.audio.n_mels),
            n_fft=saved_cfg.get("n_fft", config.audio.n_fft),
            hop_length=saved_cfg.get("hop_length", config.audio.hop_length),
            target_length=saved_cfg.get("target_length", config.audio.target_length),
        )

        self.transform = MelSpectrogramTransform(audio_config)

        self.model = AudioQualityCNN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "모델 로드 완료: {} (epoch={}, val_acc={:.3f})",
            model_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_acc", 0.0),
        )

    def predict(self, file_path: str, threshold: float = 0.5) -> tuple[bool, float]:
        """단일 파일 품질 예측.

        Returns:
            (is_good, confidence) — is_good이 True이면 정상 오디오.
            confidence는 good일 확률 (0~1).
        """
        waveform, sr = torchaudio.load(file_path)
        mel = self.transform(waveform, sr)  # (1, n_mels, target_length)
        mel = mel.unsqueeze(0).to(self.device)  # (1, 1, n_mels, target_length)

        with torch.no_grad():
            logit = self.model(mel)
            prob_bad = torch.sigmoid(logit).item()

        prob_good = 1.0 - prob_bad
        is_good = prob_good >= threshold
        return is_good, prob_good
