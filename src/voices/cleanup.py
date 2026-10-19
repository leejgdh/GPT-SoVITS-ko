"""학습 산출물 정리 — 삭제 대상을 계획하고 실행한다. voice.yaml 의 가중치와 감정 기준 음성만 보존한다."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from loguru import logger

from src.config.voice import load_voice_profile
from src.exceptions import CleanupError

_ARTIFACT_DIRS = ("step1", "step2", "step3", "logs_s1", "logs_s2")
_ASR_DIR = "04_asr"
_RAW_AUDIO_DIR = "raw_audio"


@dataclass(frozen=True)
class CleanupTarget:
    """삭제할 파일 또는 폴더(절대 경로)와 그 용량(바이트)."""

    path: str
    bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    """정리 계획. ``preserve`` 는 보존할 절대 경로, ``targets`` 는 삭제 대상."""

    preserve: list[str]
    targets: list[CleanupTarget]

    @property
    def total_bytes(self) -> int:
        return sum(t.bytes for t in self.targets)


def human_size(n: float) -> str:
    """바이트 수를 사람이 읽는 단위로."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _size_bytes(path: str) -> int:
    """파일 또는 폴더의 총 크기. 접근하지 못하는 항목은 0 으로 센다."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def _preserve_paths(voice_dir: str) -> set[str]:
    """voice.yaml 에서 보존할 절대 경로(가중치 + 감정 기준 음성)를 모은다."""
    profile = load_voice_profile(voice_dir)
    if profile is None:
        raise CleanupError(f"voice.yaml 이 없어 정리 기준을 정할 수 없습니다: {voice_dir}")

    paths: set[str] = set()
    if profile.gpt_weights:
        paths.add(os.path.abspath(profile.gpt_weights))
    if profile.sovits_weights:
        paths.add(os.path.abspath(profile.sovits_weights))
    for emo in profile.emotions.values():
        if emo.ref_audio:
            paths.add(os.path.abspath(emo.ref_audio))
    return paths


def _has_preserved_inside(dir_abs: str, preserve: set[str]) -> bool:
    prefix = dir_abs + os.sep
    return any(keep.startswith(prefix) for keep in preserve)


def _collect_unpreserved_files(dir_abs: str, preserve: set[str], targets: list[str]) -> None:
    for root, _, files in os.walk(dir_abs):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.abspath(fp) not in preserve:
                targets.append(fp)


def _step1_targets(step1_dir: str, preserve: set[str], targets: list[str]) -> None:
    """step1 에서 ``04_asr`` 만 남기고 나머지를 모은다."""
    for entry in sorted(os.listdir(step1_dir)):
        entry_path = os.path.join(step1_dir, entry)
        if entry == _ASR_DIR:
            continue
        if os.path.isfile(entry_path):
            if os.path.abspath(entry_path) not in preserve:
                targets.append(entry_path)
        elif _has_preserved_inside(os.path.abspath(entry_path), preserve):
            _collect_unpreserved_files(entry_path, preserve, targets)
        else:
            targets.append(entry_path)


def _target_paths(voice_dir: str, preserve: set[str], keep_raw: bool, keep_asr: bool) -> list[str]:
    """삭제 대상 절대 경로. 보존 파일이 안에 있는 폴더는 파일 단위로, 아니면 폴더 통째로 모은다."""
    targets: list[str] = []
    voice_abs = os.path.abspath(voice_dir)

    candidates = list(_ARTIFACT_DIRS)
    if not keep_raw:
        candidates.append(_RAW_AUDIO_DIR)

    for sub in candidates:
        full = os.path.join(voice_abs, sub)
        if not os.path.exists(full):
            continue
        if sub == "step1" and keep_asr:
            _step1_targets(full, preserve, targets)
        elif _has_preserved_inside(os.path.abspath(full), preserve):
            _collect_unpreserved_files(full, preserve, targets)
        else:
            targets.append(full)
    return targets


def _drop_nested(targets: list[str]) -> list[str]:
    """폴더가 통째로 대상이면 그 안의 파일은 목록에서 뺀다."""
    dirs = {t for t in targets if os.path.isdir(t)}
    return [t for t in targets if not (os.path.isfile(t) and any(t.startswith(d + os.sep) for d in dirs))]


def plan_cleanup(voice_dir: str, *, keep_raw: bool, keep_asr: bool) -> CleanupPlan:
    """정리 계획을 만든다. voice.yaml 이 없으면 ``CleanupError``.

    삭제 대상: step1 · step2 · step3 · logs_s1 · logs_s2, 그리고 ``keep_raw`` 가 아니면 raw_audio.
    ``keep_asr`` 이면 step1/04_asr 은 남긴다.
    """
    if not os.path.isdir(voice_dir):
        raise CleanupError(f"voice 폴더가 없습니다: {voice_dir}")
    preserve = _preserve_paths(voice_dir)
    targets = _drop_nested(_target_paths(voice_dir, preserve, keep_raw, keep_asr))
    return CleanupPlan(
        preserve=sorted(preserve),
        targets=[CleanupTarget(path=t, bytes=_size_bytes(t)) for t in sorted(targets)],
    )


def _prune_empty_dirs(voice_dir: str) -> None:
    voice_abs = os.path.abspath(voice_dir)
    for root, dirs, files in os.walk(voice_abs, topdown=False):
        if os.path.abspath(root) == voice_abs:
            continue
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass


def execute_cleanup(voice_dir: str, plan: CleanupPlan) -> None:
    """계획의 삭제 대상을 지우고 빈 폴더를 정리한다. 지우지 못한 항목은 경고만 남긴다."""
    for target in plan.targets:
        try:
            if os.path.isdir(target.path):
                shutil.rmtree(target.path)
            elif os.path.exists(target.path):
                os.remove(target.path)
        except OSError as exc:
            logger.warning("삭제 실패 ({}): {}", target.path, exc)
    _prune_empty_dirs(voice_dir)
    logger.info("정리 완료 — 해제: {}", human_size(plan.total_bytes))
