"""job 종류 — 실행할 명령과 그 params 모델을 등록한다."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Version = Literal["v2", "v3", "v4", "v2Pro", "v2ProPlus"]
RefLang = Literal["ko", "en", "ja"]
Step = Literal["step1", "step2", "step3", "step4"]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoParams(_Params):
    """params 가 없는 종류."""


class VersionParams(_Params):
    version: Version


class TrainParams(VersionParams):
    epochs: int | None = Field(None, gt=0)
    batch_size: int | None = Field(None, gt=0)


class InferParams(VersionParams):
    output_text: str = Field(min_length=1)
    ref_audio: str | None = None
    ref_text: str | None = None
    ref_lang: RefLang | None = None


class PipelineParams(TrainParams, InferParams):
    skip: list[Step] = Field(default_factory=list)


@dataclass(frozen=True)
class JobKind:
    """한 종류의 job. ``entry`` 는 프로젝트 루트 기준 실행 인자 앞부분이다.

    ``voice_paths`` 에 적은 params 필드는 voice 폴더 기준 상대 경로로 받아 실행 인자로 풀어 준다.
    ``serverless`` 이면 학습 명령이 띄우는 백그라운드 서버를 끄는 ``--no-server`` 를 붙인다.
    """

    entry: tuple[str, ...]
    params: type[_Params]
    serverless: bool = True
    voice_paths: frozenset[str] = frozenset()


def _main(command: str, params: type[_Params], voice_paths: frozenset[str] = frozenset()) -> JobKind:
    return JobKind(entry=("main.py", command), params=params, voice_paths=voice_paths)


_REF_AUDIO = frozenset({"ref_audio"})


KINDS: dict[str, JobKind] = {
    "pipeline": _main("pipeline", PipelineParams, _REF_AUDIO),
    "step1": _main("step1", NoParams),
    "step2": _main("step2", VersionParams),
    "step3": _main("step3", TrainParams),
    "step4": _main("step4", InferParams, _REF_AUDIO),
    "denoise": _main("denoise", NoParams),
    "slice": _main("slice", NoParams),
    "uvr5": _main("uvr5", NoParams),
    "asr": _main("asr", NoParams),
    "classify": _main("classify", NoParams),
    "get-text": _main("get-text", VersionParams),
    "get-hubert": _main("get-hubert", VersionParams),
    "get-sv": _main("get-sv", VersionParams),
    "get-semantic": _main("get-semantic", VersionParams),
    "train-gpt": _main("train-gpt", TrainParams),
    "train-sovits": _main("train-sovits", TrainParams),
    "vc-train": JobKind(
        entry=("tools/voice-checker/main.py", "train", "-c", "config.yaml"),
        params=NoParams,
        serverless=False,
    ),
}


def build_argv(kind: JobKind, voice_dir: str, params: _Params) -> list[str]:
    """job 을 실행할 명령줄. 값이 없는 필드는 빼서 CLI 가 자기 기본을 쓰게 한다."""
    argv = [sys.executable, *kind.entry, "--voice-dir", voice_dir]
    if kind.serverless:
        argv.append("--no-server")
    for name in type(params).model_fields:
        value = getattr(params, name)
        if value is None or value == []:
            continue
        flag = "--" + name.replace("_", "-")
        if name in kind.voice_paths:
            value = os.path.join(voice_dir, value)
        if isinstance(value, list):
            argv += [flag, *value]
        else:
            argv += [flag, str(value)]
    return argv
