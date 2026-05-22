from __future__ import annotations

import asyncio
import gc
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

    @classmethod
    def create(cls, config: Config) -> ServiceContext:
        """설정에서 TTS 파이프라인을 초기화하고 voice를 스캔한다."""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        from src.config.config import detect_device, detect_half

        ctx = cls(config)

        # device / is_half 가 conf.yaml 에 명시되지 않았으면 자동 감지로 채운다.
        # (TTS_Config 기본값은 device='cpu', is_half=False 라 GPU 가 있어도 못 활용.)
        # custom 키를 만들면 TTS_Config 가 v2Pro default 를 무시하므로 — version /
        # weights 경로가 누락되어 assert 실패. v2Pro default 를 base 로 깔고 사용자
        # override 를 위에 머지한다.
        tts_dict = dict(config.tts) if isinstance(config.tts, dict) else {}
        user_custom = dict(tts_dict.get("custom", {}))
        custom = {**TTS_Config.default_configs["v2Pro"], **user_custom}
        if "device" not in user_custom:
            custom["device"] = str(detect_device())
        if "is_half" not in user_custom:
            custom["is_half"] = detect_half()
        tts_dict["custom"] = custom

        # TTS 파이프라인 초기화 (pretrained 모델 없으면 스킵)
        try:
            ctx._tts_config = TTS_Config(tts_dict)
            logger.info(
                "TTS device={} is_half={} version={}",
                ctx._tts_config.device, ctx._tts_config.is_half, ctx._tts_config.version,
            )
            ctx._tts_pipeline = TTS(ctx._tts_config)
            logger.info("TTS 파이프라인 초기화 완료")
        except (FileNotFoundError, OSError) as e:
            logger.warning("TTS 파이프라인 초기화 스킵 (모델 미설치): {}", e)
            logger.warning("TTS 합성 비활성 — 라벨링/검수 기능만 사용 가능")

        # voice 스캔
        ctx._voices = scan_voices(config.voices_dir)

        # 기본 voice 로드 (available 상태인 경우만)
        if config.default_voice and ctx._tts_pipeline is not None:
            profile = ctx._voices.get(config.default_voice)
            if profile and profile.available:
                ctx.switch_voice(config.default_voice)
            elif profile and not profile.available:
                logger.info("기본 voice '{}' 학습 미완료 (available: false)", config.default_voice)
            else:
                logger.warning("기본 voice '{}' 를 찾을 수 없습니다", config.default_voice)

        if config.voice_checker is not None:
            logger.info("Voice Checker 활성화")

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
        from src.metrics import active_voice, voice_switch_total

        if name not in self._voices:
            from src.exceptions import VoiceNotFoundError
            msg = f"voice '{name}' 이(가) 등록되어 있지 않습니다"
            raise VoiceNotFoundError(msg)

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

        # 이전 voice 의 가중치 메모리 회수 — init_*_weights 가 self.{t2s,vits}_model
        # 을 *덮어쓰기* 라 옛 모델은 reference 0 인데 PyTorch CUDA cache 가 잡고
        # 있다. 명시 gc + cache flush 로 CPU RAM / VRAM 양쪽 회수.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("CUDA empty_cache skip ({}): {}", type(e).__name__, e)

        # 메트릭: 이전 voice 를 0 으로 내리고 새 voice 를 1 로.
        if self._current_voice:
            active_voice.labels(voice=self._current_voice).set(0)
        active_voice.labels(voice=name).set(1)
        voice_switch_total.labels(voice=name).inc()

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
