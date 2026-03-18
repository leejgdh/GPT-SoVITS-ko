# -*- coding: utf-8 -*-
"""UVR5 보컬 분리 CLI.

지정 폴더의 오디오 파일에서 보컬을 분리한다.
지원 모델: VR(HP2/HP3/HP5/DeEcho), MDXNet(onnx_dereverb), BSRoformer

출력:
  - {save-vocal-dir}/ (보컬 오디오)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import traceback

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

project_root = setup_paths()

# UVR5 내부 모듈 import를 위한 경로 추가
_uvr5_dir = str(project_root / "tools" / "uvr5")
if _uvr5_dir not in sys.path:
    sys.path.insert(0, _uvr5_dir)
# ---------------------------------------------------------------

import ffmpeg
import torch
from bsroformer import Roformer_Loader
from loguru import logger
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho

from tools.utils.download import ensure_dir, ensure_file
from tools.utils.audio import clean_path

_HF_REPO_ID = "lj1995/VoiceConversionWebUI"
_HF_SUBFOLDER = "uvr5_weights"

# 축약 별칭 → HF 원본 모델명 (확장자 제외)
_MODEL_ALIAS: dict[str, str] = {
    "HP2": "HP2_all_vocals",
    "HP3": "HP3_all_vocals",
    "HP5": "HP5_only_main_vocal",
    "DeEcho-Aggressive": "VR-DeEchoAggressive",
    "DeEcho-Normal": "VR-DeEchoNormal",
    "DeEcho-DeReverb": "VR-DeEchoDeReverb",
}
_DEFAULT_WEIGHT_DIR = "data/models/uvr5"


def _list_available_models(weight_dir: str) -> list[str]:
    """가중치 디렉토리에서 사용 가능한 모델 이름 목록을 반환한다."""
    names = []
    if not os.path.isdir(weight_dir):
        return names
    for name in os.listdir(weight_dir):
        if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
            names.append(name.replace(".pth", "").replace(".ckpt", ""))
    return names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UVR5 보컬 분리")
    parser.add_argument("--voice-dir", default=None, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument("--model-name", default="HP5_only_main_vocal", help="UVR5 모델 이름 (기본: HP5_only_main_vocal)")
    parser.add_argument("--inp-dir", default=None, help="입력 오디오 폴더 (기본: {voice-dir}/step1/02_sliced)")
    parser.add_argument("--save-vocal-dir", default=None, help="보컬 출력 폴더 (기본: {voice-dir}/step1/03_vocal)")
    parser.add_argument(
        "--weight-dir",
        default=_DEFAULT_WEIGHT_DIR,
        help=f"UVR5 가중치 폴더 (기본: {_DEFAULT_WEIGHT_DIR})",
    )
    parser.add_argument("--agg", type=int, default=10, help="인공물 제거 강도 (VR 모델용, 기본: 10)")
    parser.add_argument(
        "--format", default="flac", choices=["wav", "flac", "mp3", "m4a"],
        help="출력 오디오 포맷 (기본: flac)",
    )
    parser.add_argument("--device", default=None, help="디바이스 (기본: 자동 감지)")
    parser.add_argument("--no-half", action="store_true", help="FP32 사용 (기본: FP16)")
    parser.add_argument("--list-models", action="store_true", help="사용 가능한 모델 목록 출력 후 종료")
    args = parser.parse_args()

    if not args.list_models:
        if args.voice_dir is None and (args.inp_dir is None or args.save_vocal_dir is None):
            parser.error("--voice-dir 또는 --inp-dir/--save-vocal-dir 를 지정해야 합니다")
        if args.inp_dir is None:
            args.inp_dir = os.path.join(args.voice_dir, "step1", "02_sliced")
        if args.save_vocal_dir is None:
            args.save_vocal_dir = os.path.join(args.voice_dir, "step1", "03_vocal")

    return args


def _reformat_audio(inp_path: str, tmp_dir: str) -> str | None:
    """44100Hz 스테레오가 아닌 오디오를 임시 파일로 변환한다.

    Args:
        inp_path: 입력 오디오 파일 경로.
        tmp_dir: 변환된 임시 파일을 저장할 디렉토리 경로.
    """
    try:
        info = ffmpeg.probe(inp_path, cmd="ffprobe")
        stream = info["streams"][0]
        if stream["channels"] == 2 and stream["sample_rate"] == "44100":
            return None
    except Exception:
        pass

    tmp_path = os.path.join(tmp_dir, f"{os.path.basename(inp_path)}.reformatted.wav")
    subprocess.run(
        [
            "ffmpeg", "-i", inp_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ac", "2", "-ar", "44100",
            tmp_path, "-y",
        ],
        check=True,
    )
    return tmp_path


def main() -> None:
    args = _parse_args()

    if args.list_models:
        models = _list_available_models(args.weight_dir)
        if models:
            logger.info("사용 가능한 모델:")
            for m in sorted(models):
                logger.info("  - {}", m)
        else:
            logger.warning("가중치 디렉토리에 모델이 없습니다: {}", args.weight_dir)
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    is_half = (not args.no_half) and torch.cuda.is_available()

    inp_dir = clean_path(args.inp_dir)
    save_vocal_dir = clean_path(args.save_vocal_dir)

    os.makedirs(save_vocal_dir, exist_ok=True)

    # 별칭이면 HF 원본 이름으로 변환
    model_name = _MODEL_ALIAS.get(args.model_name, args.model_name)
    weight_dir = args.weight_dir
    is_hp3 = "HP3" in model_name

    # 가중치 자동 다운로드 + 모델 로딩
    if model_name == "onnx_dereverb_By_FoxJoy":
        onnx_dir = os.path.join(weight_dir, "onnx_dereverb_By_FoxJoy")
        ensure_dir(
            onnx_dir, _HF_REPO_ID,
            allow_patterns=[f"{_HF_SUBFOLDER}/onnx_dereverb_By_FoxJoy/*"],
        )
        pre_fun = MDXNetDereverb(15)
    elif "roformer" in model_name.lower():
        ckpt_path = os.path.join(weight_dir, f"{model_name}.ckpt")
        ensure_file(
            ckpt_path, _HF_REPO_ID,
            filename=f"{model_name}.ckpt", subfolder=_HF_SUBFOLDER,
        )
        config_path = os.path.join(weight_dir, f"{model_name}.yaml")
        if not os.path.exists(config_path):
            logger.warning(
                "설정 파일 없음: {}\n"
                "기본 설정을 사용합니다. 일부 모델에서는 동작하지 않을 수 있습니다.\n"
                "'{}/{}' 파일을 직접 배치해 주세요.",
                config_path, weight_dir, f"{model_name}.yaml",
            )
        pre_fun = Roformer_Loader(
            model_path=ckpt_path,
            config_path=config_path,
            device=device,
            is_half=is_half,
        )
    else:
        pth_path = os.path.join(weight_dir, f"{model_name}.pth")
        ensure_file(
            pth_path, _HF_REPO_ID,
            filename=f"{model_name}.pth", subfolder=_HF_SUBFOLDER,
        )
        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=args.agg,
            model_path=pth_path,
            device=device,
            is_half=is_half,
        )

    # 오디오 파일 처리
    paths = [os.path.join(inp_dir, name) for name in os.listdir(inp_dir)]
    with tempfile.TemporaryDirectory(prefix="uvr5_reformat_") as tmp_dir:
        for inp_path in paths:
            if not os.path.isfile(inp_path):
                continue
            try:
                reformatted = _reformat_audio(inp_path, tmp_dir)
                actual_path = reformatted or inp_path
                pre_fun._path_audio_(actual_path, None, save_vocal_dir, args.format, is_hp3)
                logger.info("{} -> 성공", os.path.basename(inp_path))
            except Exception as e:
                logger.warning("{} -> 건너뜀 ({})", os.path.basename(inp_path), e)
                logger.debug("상세 traceback:\n{}", traceback.format_exc())

    # 정리
    try:
        if model_name == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
            del pre_fun
    except Exception as e:
        logger.warning("모델 정리 중 오류 ({})", e)
        logger.debug("상세 traceback:\n{}", traceback.format_exc())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
