"""tts-service 예외 계층.

도메인별 하위 예외를 둔다. 내장 예외(ValueError/RuntimeError/KeyError 등)
대신 여기 정의된 예외를 사용한다.
"""

from __future__ import annotations


class TTSServiceError(Exception):
    """tts-service 기본 예외."""


class ConfigError(TTSServiceError):
    """설정 로드/파싱 실패."""


class VoiceNotFoundError(TTSServiceError):
    """등록되지 않은 voice 요청."""


class VoiceLoadError(TTSServiceError):
    """voice 가중치 로드 실패."""


class SynthesisError(TTSServiceError):
    """합성 파이프라인 실행 실패."""


class VoiceExistsError(TTSServiceError):
    """이미 있는 이름으로 voice 를 만들려 함."""


class UnsupportedFileError(TTSServiceError):
    """그 단계가 받지 않는 확장자의 파일을 올리려 함."""


class StageFileExistsError(TTSServiceError):
    """같은 이름의 파일이 단계 폴더에 이미 있음."""


class InvalidFileNameError(TTSServiceError):
    """라벨 목록을 깨뜨리는 문자가 파일 이름에 들어 있음."""


class InvalidLabelListError(TTSServiceError):
    """올린 음성 인식 목록의 형식이 맞지 않거나 없는 음성을 가리킴."""


class CleanupError(TTSServiceError):
    """학습 산출물 정리 기준을 정할 수 없음."""


__all__ = [
    "TTSServiceError",
    "ConfigError",
    "VoiceNotFoundError",
    "VoiceLoadError",
    "SynthesisError",
    "VoiceExistsError",
    "UnsupportedFileError",
    "StageFileExistsError",
    "InvalidFileNameError",
    "InvalidLabelListError",
    "CleanupError",
]
