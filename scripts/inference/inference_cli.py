"""GPT-SoVITS CLI 추론.

TTS_infer_pack.TTS를 직접 사용하여 텍스트 → 음성 합성을 수행한다.
gradio 의존성 없이 독립 실행 가능.

사용 예 (--voice-dir):
  uv run python scripts/inference/inference_cli.py \
    --voice-dir data/voice/lunabi \
    --ref-audio data/voice/lunabi/step1/03_vocal/sample.flac \
    --ref-text "참조 텍스트" \
    --text "합성할 텍스트"

사용 예 (수동 지정):
  uv run python scripts/inference/inference_cli.py \
    --gpt-weights model.ckpt --sovits-weights model.pth \
    --ref-audio ref.wav --ref-text "참조 텍스트" \
    --text "합성할 텍스트" --output result.wav
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# ---------------------------------------------------------------

import soundfile as sf
import yaml
from loguru import logger

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-SoVITS CLI 추론")
    parser.add_argument(
        "--voice-dir", default=None,
        help="캐릭터 음성 폴더 (예: data/voice/lunabi). "
             "step3에서 최신 가중치를 자동 탐색하고, 출력을 step4/에 저장한다.",
    )
    parser.add_argument(
        "--config", default=None,
        help="TTS 설정 파일 경로 (미지정시 기본 설정 사용)",
    )
    parser.add_argument("--ref-audio", required=True, help="참조 오디오 파일 경로")
    parser.add_argument("--ref-text", required=True, help="참조 텍스트 (직접 입력 또는 @파일경로)")
    parser.add_argument("--ref-lang", default="ko", choices=["ko", "en", "ja"], help="참조 언어 (기본: ko)")
    parser.add_argument("--text", required=True, help="합성할 텍스트 (직접 입력 또는 @파일경로)")
    parser.add_argument("--text-lang", default="ko", choices=["ko", "en", "ja"], help="합성 텍스트 언어 (기본: ko)")
    parser.add_argument("--output", default=None, help="출력 파일 경로 (--voice-dir 시 자동 설정)")
    parser.add_argument(
        "--version", default="v2Pro",
        choices=["v2", "v3", "v4", "v2Pro", "v2ProPlus"],
        help="모델 버전 (기본: v2Pro). step3/{version}/ 하위에서 가중치를 탐색한다.",
    )
    parser.add_argument("--gpt-weights", default=None, help="GPT 가중치 경로 (--voice-dir 시 자동 탐색)")
    parser.add_argument("--sovits-weights", default=None, help="SoVITS 가중치 경로 (--voice-dir 시 자동 탐색)")
    parser.add_argument("--speed", type=float, default=1.0, help="속도 배율")
    parser.add_argument("--top-k", type=int, default=15, help="Top-K 샘플링")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-P 샘플링")
    parser.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도")
    parser.add_argument("--seed", type=int, default=-1, help="랜덤 시드 (-1=랜덤)")
    parser.add_argument("--text-split-method", default="cut5", help="텍스트 분할 방법")
    return parser.parse_args()


def _find_latest_weights(directory: str, pattern: str) -> str | None:
    """디렉토리에서 가장 최근 수정된 가중치 파일을 찾는다."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _resolve_voice_dir(args: argparse.Namespace) -> None:
    """--voice-dir에서 가중치 경로와 출력 경로를 자동 설정한다."""
    voice_dir = args.voice_dir
    step3 = os.path.join(voice_dir, "step3", args.version)

    if args.gpt_weights is None:
        gpt_dir = os.path.join(step3, "02_gpt_weights")
        gpt_path = _find_latest_weights(gpt_dir, "*.ckpt")
        if gpt_path is None:
            logger.error("GPT 가중치를 찾을 수 없습니다: {}", gpt_dir)
            sys.exit(1)
        args.gpt_weights = gpt_path
        logger.info("GPT 가중치: {}", gpt_path)

    if args.sovits_weights is None:
        sovits_dir = os.path.join(step3, "04_sovits_weights")
        sovits_path = _find_latest_weights(sovits_dir, "*.pth")
        if sovits_path is None:
            logger.error("SoVITS 가중치를 찾을 수 없습니다: {}", sovits_dir)
            sys.exit(1)
        args.sovits_weights = sovits_path
        logger.info("SoVITS 가중치: {}", sovits_path)

    if args.output is None:
        step4 = os.path.join(voice_dir, "step4", args.version)
        os.makedirs(step4, exist_ok=True)
        args.output = os.path.join(step4, "output.wav")


def _read_text(value: str) -> str:
    """'@파일경로' 형식이면 파일 내용을 읽고, 아니면 문자열 그대로 반환."""
    if value.startswith("@") and os.path.isfile(value[1:]):
        with open(value[1:], "r", encoding="utf-8") as f:
            return f.read().strip()
    return value


def main() -> None:
    args = _parse_args()

    if args.voice_dir:
        _resolve_voice_dir(args)
    elif args.output is None:
        args.output = "output.wav"

    if args.gpt_weights is None and args.sovits_weights is None and args.voice_dir is None:
        logger.error("--voice-dir 또는 --gpt-weights/--sovits-weights를 지정하세요.")
        sys.exit(1)

    ref_text = _read_text(args.ref_text)
    target_text = _read_text(args.text)

    # TTS 파이프라인 초기화
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        tts_config = TTS_Config(data.get("tts", {}), config_save_path=args.config)
    else:
        tts_config = TTS_Config()
    tts = TTS(tts_config)

    if args.gpt_weights:
        tts.init_t2s_weights(args.gpt_weights)
    if args.sovits_weights:
        tts.init_vits_weights(args.sovits_weights)

    req = {
        "text": target_text,
        "text_lang": args.text_lang,
        "ref_audio_path": args.ref_audio,
        "prompt_text": ref_text,
        "prompt_lang": args.ref_lang,
        "speed_factor": args.speed,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "seed": args.seed,
        "text_split_method": args.text_split_method,
    }

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 합성 실행
    logger.info("합성 시작: \"{}\"", target_text[:50] + "..." if len(target_text) > 50 else target_text)
    for sr, audio_data in tts.synthesize(req):
        sf.write(args.output, audio_data, sr)
        logger.info("오디오 저장 완료: {} ({:.1f}초)", args.output, len(audio_data)/sr)
        break


if __name__ == "__main__":
    main()
