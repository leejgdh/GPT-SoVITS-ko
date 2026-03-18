"""Voice Checker 설정 — 루트 Config에서 re-export.

voice-checker 내부 코드는 이 모듈에서 import하며,
실제 정의는 프로젝트 루트 src/config/config.py에 있다.
"""
import importlib.util
import sys
from pathlib import Path

# 루트 src/config/config.py를 직접 로드 (모듈명 충돌 회피)
_root_config_path = Path(__file__).resolve().parents[4] / "src" / "config" / "config.py"
if not _root_config_path.exists():
    raise FileNotFoundError(f"루트 config.py를 찾을 수 없습니다: {_root_config_path}")
_spec = importlib.util.spec_from_file_location("_root_config", str(_root_config_path))
_root_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _root_config
_spec.loader.exec_module(_root_config)

AudioConfig = _root_config.VCAudioConfig
AugmentationConfig = _root_config.VCAugmentationConfig
InferenceConfig = _root_config.VCInferenceConfig
TrainingConfig = _root_config.VCTrainingConfig
Config = _root_config.VoiceCheckerConfig

__all__ = [
    "AudioConfig",
    "AugmentationConfig",
    "InferenceConfig",
    "TrainingConfig",
    "Config",
]
