"""Voice 프로필 관리.

voice.yaml 스키마:
  name: lunabi
  version: v2Pro
  ref_lang: ko
  gpt_weights: step3/v2Pro/02_gpt_weights/model.ckpt
  sovits_weights: step3/v2Pro/04_sovits_weights/model.pth
  emotions:
    default:
      ref_audio: step1/03_vocal/normal_001.flac
      ref_text: "평범한 톤의 참조 텍스트"
    happy:
      ref_audio: step1/03_vocal/happy_003.flac
      ref_text: "기쁜 톤의 참조 텍스트"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml
from loguru import logger

_VOICE_YAML = "voice.yaml"


@dataclass
class EmotionRef:
    """감정별 레퍼런스 오디오/텍스트."""

    ref_audio: str
    ref_text: str


@dataclass
class VoiceProfile:
    """캐릭터 음성 프로필."""

    name: str
    version: str = ""
    ref_lang: str = "ko"
    gpt_weights: str = ""
    sovits_weights: str = ""
    available: bool = False
    emotions: dict[str, EmotionRef] = field(default_factory=dict)

    def get_emotion(self, emotion: str | None = None) -> EmotionRef:
        """감정 이름으로 EmotionRef를 반환한다.

        폴백 순서: 지정 감정 → default → 첫 번째 항목.
        """
        if emotion and emotion in self.emotions:
            return self.emotions[emotion]
        if "default" in self.emotions:
            return self.emotions["default"]
        return next(iter(self.emotions.values()))

    @property
    def ref_audio(self) -> str:
        """하위 호환용 — default 감정의 ref_audio."""
        return self.get_emotion().ref_audio

    @property
    def ref_text(self) -> str:
        """하위 호환용 — default 감정의 ref_text."""
        return self.get_emotion().ref_text

    @property
    def emotion_names(self) -> list[str]:
        """등록된 감정 이름 목록."""
        return list(self.emotions.keys())


def load_voice_profile(voice_dir: str) -> VoiceProfile | None:
    """voice.yaml을 로드하고 상대경로를 절대경로로 변환한다.

    Returns:
        VoiceProfile 또는 voice.yaml이 없으면 None.
    """
    yaml_path = os.path.join(voice_dir, _VOICE_YAML)
    if not os.path.isfile(yaml_path):
        return None

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        logger.warning("voice.yaml이 비어 있습니다: {}", yaml_path)
        return None

    if "name" not in data:
        logger.warning("voice.yaml에 name 키가 없습니다: {}", yaml_path)
        return None

    available = data.get("available", False)

    # emotions 파싱 (하위 호환: 기존 ref_audio/ref_text → emotions.default)
    emotions: dict[str, EmotionRef] = {}
    if "emotions" in data and isinstance(data["emotions"], dict):
        for emo_name, emo_data in data["emotions"].items():
            if isinstance(emo_data, dict) and "ref_audio" in emo_data:
                emotions[emo_name] = EmotionRef(
                    ref_audio=os.path.join(voice_dir, emo_data["ref_audio"]),
                    ref_text=emo_data.get("ref_text", ""),
                )
    elif "ref_audio" in data:
        emotions["default"] = EmotionRef(
            ref_audio=os.path.join(voice_dir, data["ref_audio"]),
            ref_text=data.get("ref_text", ""),
        )

    return VoiceProfile(
        name=data["name"],
        version=data.get("version", ""),
        ref_lang=data.get("ref_lang", "ko"),
        gpt_weights=os.path.join(voice_dir, data["gpt_weights"]) if data.get("gpt_weights") else "",
        sovits_weights=os.path.join(voice_dir, data["sovits_weights"]) if data.get("sovits_weights") else "",
        available=available,
        emotions=emotions,
    )


def scan_voices(voices_dir: str) -> dict[str, VoiceProfile]:
    """voices_dir 하위에서 voice.yaml이 있는 디렉토리를 모두 스캔한다.

    Returns:
        {name: VoiceProfile} 딕셔너리.
    """
    profiles: dict[str, VoiceProfile] = {}

    if not os.path.isdir(voices_dir):
        logger.warning("voices 디렉토리가 존재하지 않습니다: {}", voices_dir)
        return profiles

    for entry in sorted(os.listdir(voices_dir)):
        entry_path = os.path.join(voices_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        profile = load_voice_profile(entry_path)
        if profile is not None:
            profiles[profile.name] = profile
            logger.debug("voice 로드: {} (version={})", profile.name, profile.version)

    logger.info("voice {} 개 로드 완료", len(profiles))
    return profiles


def save_voice_yaml(
    voice_dir: str,
    *,
    name: str,
    version: str,
    ref_audio: str,
    ref_text: str,
    ref_lang: str,
    gpt_weights: str,
    sovits_weights: str,
    emotions: dict[str, dict[str, str]] | None = None,
) -> str:
    """voice.yaml을 생성한다. 경로는 voice_dir 기준 상대경로로 저장.

    Returns:
        생성된 voice.yaml의 절대경로.
    """
    yaml_path = os.path.join(voice_dir, _VOICE_YAML)

    # emotions가 없으면 ref_audio/ref_text로 default 생성
    if not emotions:
        emotions = {
            "default": {
                "ref_audio": os.path.relpath(ref_audio, voice_dir),
                "ref_text": ref_text,
            },
        }

    data = {
        "name": name,
        "version": version,
        "ref_lang": ref_lang,
        "gpt_weights": os.path.relpath(gpt_weights, voice_dir),
        "sovits_weights": os.path.relpath(sovits_weights, voice_dir),
        "available": True,
        "emotions": emotions,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info("voice.yaml 저장: {}", yaml_path)
    return yaml_path
