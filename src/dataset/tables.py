"""step2 산출물(음소 · semantic 토큰)과 학습 곡선을 행 · 점 단위로 읽는다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_SEMANTIC_PREVIEW_TOKENS = 20
_MAX_CURVE_POINTS = 500


@dataclass(frozen=True)
class TextRow:
    """``name2text.txt`` 한 줄 — 파일 이름, 음소열, 단어별 음소 수, 원문."""

    name: str
    phonemes: str
    word2ph: str
    text: str


@dataclass(frozen=True)
class SemanticRow:
    """``name2semantic.tsv`` 한 줄 — 파일 이름, 토큰 수, 앞쪽 토큰 일부."""

    name: str
    token_count: int
    preview: list[int]


@dataclass(frozen=True)
class CurveSeries:
    """한 학습 로그의 스칼라 하나. ``points`` 는 (step, value)."""

    source: str
    tag: str
    points: list[tuple[int, float]]


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_text_rows(path: Path, offset: int, limit: int) -> tuple[int, list[TextRow]]:
    """음소 파일의 한 페이지와 전체 줄 수."""
    lines = _read_lines(path)
    rows: list[TextRow] = []
    for line in lines[offset:offset + limit]:
        parts = line.split("\t")
        parts += [""] * (4 - len(parts))
        rows.append(TextRow(name=parts[0], phonemes=parts[1], word2ph=parts[2], text=parts[3]))
    return len(lines), rows


def read_semantic_rows(path: Path, offset: int, limit: int) -> tuple[int, list[SemanticRow]]:
    """semantic 토큰 파일의 한 페이지와 전체 줄 수."""
    lines = _read_lines(path)
    rows: list[SemanticRow] = []
    for line in lines[offset:offset + limit]:
        name, _, tokens = line.partition("\t")
        values = [int(t) for t in tokens.split() if t.lstrip("-").isdigit()]
        rows.append(SemanticRow(name=name, token_count=len(values), preview=values[:_SEMANTIC_PREVIEW_TOKENS]))
    return len(lines), rows


def _downsample(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if len(points) <= _MAX_CURVE_POINTS:
        return points
    stride = len(points) / _MAX_CURVE_POINTS
    return [points[int(i * stride)] for i in range(_MAX_CURVE_POINTS)]


def read_curves(source: str, log_dir: Path) -> list[CurveSeries] | None:
    """폴더 아래 tensorboard event 파일의 스칼라 곡선. tensorboard 가 없으면 None.

    tensorboard 는 training extras 에만 있어 호출 시점에 가져온다.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError:
        return None

    series: list[CurveSeries] = []
    for event_file in sorted(log_dir.rglob("events.out.tfevents.*")):
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags()["scalars"]:
                points = [(e.step, float(e.value)) for e in accumulator.Scalars(tag)]
                series.append(CurveSeries(source=source, tag=tag, points=_downsample(points)))
        except Exception:
            logger.exception("학습 곡선 읽기 실패: {}", event_file)
    return series
