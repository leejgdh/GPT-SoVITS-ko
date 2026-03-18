from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from _setup_paths import setup_gpt_sovits_paths
from src.config.voice import VoiceProfile, scan_voices

if TYPE_CHECKING:
    from src.config.config import Config

# GPT_SoVITS 내부 import를 위한 경로 설정 (_setup_paths.py 단일 소스)
setup_gpt_sovits_paths()


class ServiceContext:
    """TTS 서비스 의존성 컨테이너."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._tts_pipeline = None
        self._tts_config = None
        self._voices: dict[str, VoiceProfile] = {}
        self._current_voice: str | None = None
        self._lock = asyncio.Lock()
        self._vc_data_dir: Path | None = None
        self._vc_labels_file: Path | None = None

    @classmethod
    def create(cls, config: Config) -> ServiceContext:
        """설정에서 TTS 파이프라인을 초기화하고 voice를 스캔한다."""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        ctx = cls(config)

        # TTS 파이프라인 초기화 (pretrained 모델 없으면 스킵)
        try:
            ctx._tts_config = TTS_Config(config.tts)
            ctx._tts_pipeline = TTS(ctx._tts_config)
            logger.info("TTS 파이프라인 초기화 완료")
        except (FileNotFoundError, OSError) as e:
            logger.warning("TTS 파이프라인 초기화 스킵 (모델 미설치): {}", e)
            logger.warning("TTS 합성 비활성 — 라벨링/검수 기능만 사용 가능")

        # voice 스캔
        ctx._voices = scan_voices(config.voices_dir)

        # 기본 voice 로드
        if config.default_voice:
            if config.default_voice in ctx._voices:
                ctx.switch_voice(config.default_voice)
            else:
                logger.warning(
                    "기본 voice '{}' 를 찾을 수 없습니다",
                    config.default_voice,
                )

        # voice_checker 설정이 있으면 데이터 경로 설정
        if config.voice_checker is not None:
            project_root = Path(__file__).resolve().parents[2]
            ctx._vc_data_dir = project_root / "data" / "voice-checker"
            ctx._vc_labels_file = ctx._vc_data_dir / "labels.json"
            logger.info("Voice Checker 활성화: {}", ctx._vc_data_dir)

        return ctx

    @property
    def tts(self):
        """TTS 파이프라인 인스턴스."""
        return self._tts_pipeline

    @property
    def tts_config(self):
        """TTS 설정."""
        return self._tts_config

    @property
    def voices(self) -> dict[str, VoiceProfile]:
        """등록된 voice 프로필 목록."""
        return self._voices

    @property
    def current_voice(self) -> str | None:
        """현재 로드된 voice 이름."""
        return self._current_voice

    @property
    def vc_data_dir(self) -> Path | None:
        """Voice Checker 데이터 디렉토리."""
        return self._vc_data_dir

    @property
    def vc_labels_file(self) -> Path | None:
        """Voice Checker labels.json 경로."""
        return self._vc_labels_file

    @property
    def lock(self) -> asyncio.Lock:
        """TTS 파이프라인 동시 접근 보호용 잠금."""
        return self._lock

    def warmup(self) -> None:
        """CUDA 워밍업을 위해 더미 합성을 실행한다."""
        if self._current_voice is None:
            logger.warning("워밍업 스킵: 로드된 voice 없음")
            return

        profile = self._voices[self._current_voice]
        emo = profile.get_emotion("default")
        warmup_text = {"ko": "안녕", "ja": "テスト", "en": "hello"}

        req = {
            "text": warmup_text.get(profile.ref_lang, "hello"),
            "text_lang": profile.ref_lang,
            "ref_audio_path": emo.ref_audio,
            "prompt_text": emo.ref_text,
            "prompt_lang": profile.ref_lang,
            "parallel_infer": True,
            "split_bucket": False,
            "text_split_method": "cut5",
        }

        try:
            for _sr, _audio in self._tts_pipeline.synthesize(req):
                break
            logger.info("CUDA 워밍업 완료")
        except Exception as e:
            logger.warning("CUDA 워밍업 실패: {}", e)

    def switch_voice(self, name: str) -> None:
        """voice를 전환한다. 가중치를 로드하고 현재 voice를 업데이트."""
        if name not in self._voices:
            msg = f"voice '{name}' 이(가) 등록되어 있지 않습니다"
            raise KeyError(msg)

        if name == self._current_voice:
            logger.debug("이미 '{}' voice가 로드되어 있습니다", name)
            return

        profile = self._voices[name]
        logger.info(
            "voice 전환: {} → {} (version={})",
            self._current_voice, name, profile.version,
        )

        # GPT + SoVITS 가중치 로드 (voice별로 각각 학습된 모델)
        self._tts_pipeline.init_t2s_weights(profile.gpt_weights)
        self._tts_pipeline.init_vits_weights(profile.sovits_weights)

        self._current_voice = name
        logger.info("voice '{}' 로드 완료", name)

    def get_voice_profile(self, name: str) -> VoiceProfile | None:
        """voice 프로필을 반환한다."""
        return self._voices.get(name)

    def close(self) -> None:
        """리소스 정리."""
        if self._tts_pipeline is not None:
            self._tts_pipeline.stop()
            logger.info("TTS 파이프라인 종료")
