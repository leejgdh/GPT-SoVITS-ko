"""단계 폴더의 파일 요약 · 목록 · 종류 판별."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"})
TEXT_SUFFIXES = frozenset({".txt", ".tsv", ".yaml", ".yml", ".json", ".log", ".list"})


@dataclass(frozen=True)
class StageSummary:
    """단계 폴더의 파일 수와 총 용량(바이트). 폴더가 없으면 ``exists`` 가 False."""

    exists: bool
    files: int
    bytes: int


@dataclass(frozen=True)
class FileEntry:
    """단계 폴더 안 파일 하나. ``name`` 은 단계 폴더 기준 상대 경로다."""

    name: str
    size: int
    modified: float
    duration: float | None


def _iter_files(directory: Path, recursive: bool):
    if recursive:
        for root, _dirs, names in os.walk(directory):
            for name in names:
                yield Path(root) / name
    else:
        for entry in directory.iterdir():
            if entry.is_file():
                yield entry


def summarize(directory: Path, recursive: bool) -> StageSummary:
    """폴더의 파일 수와 용량을 센다."""
    if not directory.is_dir():
        return StageSummary(exists=False, files=0, bytes=0)
    files = 0
    total = 0
    for path in _iter_files(directory, recursive):
        files += 1
        total += path.stat().st_size
    return StageSummary(exists=True, files=files, bytes=total)


def _audio_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 2)
    except Exception:
        return None


def list_files(
    directory: Path, recursive: bool, offset: int, limit: int,
) -> tuple[int, list[FileEntry]]:
    """이름순 파일 목록의 한 페이지와 전체 개수. 오디오는 길이를 함께 잰다."""
    paths = sorted(_iter_files(directory, recursive), key=lambda p: p.relative_to(directory).as_posix())
    entries: list[FileEntry] = []
    for path in paths[offset:offset + limit]:
        stat = path.stat()
        entries.append(FileEntry(
            name=path.relative_to(directory).as_posix(),
            size=stat.st_size,
            modified=stat.st_mtime,
            duration=_audio_duration(path) if path.suffix.lower() in AUDIO_SUFFIXES else None,
        ))
    return len(paths), entries
