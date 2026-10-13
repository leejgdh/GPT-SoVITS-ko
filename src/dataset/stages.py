"""열람할 수 있는 학습 단계 폴더 정의."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_VERSION_TOKEN = "{version}"


@dataclass(frozen=True)
class Stage:
    """voice 폴더 안의 한 단계 폴더. ``path`` 는 voice 폴더 기준 상대 경로다."""

    key: str
    path: str
    recursive: bool

    @property
    def versioned(self) -> bool:
        return _VERSION_TOKEN in self.path

    def directory(self, voice_dir: Path, version: str | None) -> Path:
        """단계 폴더의 실제 경로. 버전별 단계는 ``version`` 이 있어야 한다."""
        return voice_dir / self.path.replace(_VERSION_TOKEN, version or "")


STAGES: dict[str, Stage] = {
    stage.key: stage
    for stage in (
        Stage("raw_audio", "raw_audio", recursive=False),
        Stage("denoise", "step1/01_denoise", recursive=False),
        Stage("sliced", "step1/02_sliced", recursive=False),
        Stage("vocal", "step1/03_vocal", recursive=False),
        Stage("asr", "step1/04_asr", recursive=False),
        Stage("step2", "step2/{version}", recursive=False),
        Stage("hubert", "step2/{version}/01_cnhubert", recursive=False),
        Stage("wav32k", "step2/{version}/02_wav32k", recursive=False),
        Stage("speaker_vector", "step2/{version}/04_sv", recursive=False),
        Stage("step3", "step3/{version}", recursive=False),
        Stage("gpt_logs", "step3/{version}/01_gpt_logs", recursive=True),
        Stage("gpt_weights", "step3/{version}/02_gpt_weights", recursive=False),
        Stage("sovits_logs", "step3/{version}/03_sovits_logs", recursive=True),
        Stage("sovits_weights", "step3/{version}/04_sovits_weights", recursive=False),
        Stage("output", "step4/{version}", recursive=False),
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
