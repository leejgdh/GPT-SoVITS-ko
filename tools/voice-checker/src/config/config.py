"""Voice Checker 설정 — 루트 Config에서 re-export.

voice-checker 내부 코드는 이 모듈에서 import하며,
실제 정의는 프로젝트 루트 src/config/config.py에 있다.
"""
import sys
from pathlib import Path

# 루트 src/ 를 import 경로에 추가 (voice-checker 독립 실행 시 필요)
_gpt_sovits_root = str(Path(__file__).resolve().parents[4])
if _gpt_sovits_root not in sys.path:
    sys.path.insert(0, _gpt_sovits_root)

from src.config.config import (  # noqa: E402, F401
    VCAudioConfig as AudioConfig,
    VCAugmentationConfig as AugmentationConfig,
    VCInferenceConfig as InferenceConfig,
    VCTrainingConfig as TrainingConfig,
    VoiceCheckerConfig as Config,
)

__all__ = [
    "AudioConfig",
    "AugmentationConfig",
    "InferenceConfig",
    "TrainingConfig",
    "Config",
]
