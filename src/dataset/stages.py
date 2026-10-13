"""열람할 수 있는 학습 단계 폴더 정의."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.dataset.files import AUDIO_SUFFIXES

_VERSION_TOKEN = "{version}"


@dataclass(frozen=True)
class Stage:
    """voice 폴더 안의 한 단계 폴더. ``path`` 는 voice 폴더 기준 상대 경로, ``accepts`` 는 올릴 수 있는 파일 확장자(비면 올릴 수 없음)다."""

    key: str
    path: str
    recursive: bool
    accepts: frozenset[str] = frozenset()

    @property
    def versioned(self) -> bool:
        return _VERSION_TOKEN in self.path

    def directory(self, voice_dir: Path, version: str | None) -> Path:
        """단계 폴더의 실제 경로. 버전별 단계는 ``version`` 이 있어야 한다."""
        return voice_dir / self.path.replace(_VERSION_TOKEN, version or "")


_AUDIO = frozenset(AUDIO_SUFFIXES)

STAGES: dict[str, Stage] = {
    stage.key: stage
    for stage in (
        Stage("raw_audio", "raw_audio", recursive=False, accepts=_AUDIO),
        Stage("denoise", "step1/01_denoise", recursive=False, accepts=_AUDIO),
        Stage("sliced", "step1/02_sliced", recursive=False, accepts=_AUDIO),
        Stage("vocal", "step1/03_vocal", recursive=False, accepts=_AUDIO),
        Stage("asr", "step1/04_asr", recursive=False, accepts=frozenset({".list"})),
        Stage("step2", "step2/{version}", recursive=False, accepts=frozenset({".txt", ".tsv"})),
        Stage("hubert", "step2/{version}/01_cnhubert", recursive=False, accepts=frozenset({".pt"})),
        Stage("wav32k", "step2/{version}/02_wav32k", recursive=False, accepts=_AUDIO),
        Stage("speaker_vector", "step2/{version}/04_sv", recursive=False, accepts=frozenset({".pt"})),
        Stage("step3", "step3/{version}", recursive=False, accepts=frozenset({".yaml", ".yml", ".json"})),
        Stage("gpt_logs", "step3/{version}/01_gpt_logs", recursive=True),
        Stage("gpt_weights", "step3/{version}/02_gpt_weights", recursive=False, accepts=frozenset({".ckpt"})),
        Stage("sovits_logs", "step3/{version}/03_sovits_logs", recursive=True),
        Stage("sovits_weights", "step3/{version}/04_sovits_weights", recursive=False, accepts=frozenset({".pth"})),
        Stage("output", "step4/{version}", recursive=False, accepts=_AUDIO),
    )
}

_VERSION_PARENTS = ("step2", "step3", "step4")


def voice_versions(voice_dir: Path) -> list[str]:
    """step2 · step3 · step4 아래에 만들어진 모델 버전 폴더 이름."""
    found: set[str] = set()
    for parent in _VERSION_PARENTS:
        base = voice_dir / parent
        if base.is_dir():
            found.update(entry.name for entry in base.iterdir() if entry.is_dir())
    return sorted(found)
