"""학습 설정 생성 모듈.

--voice-dir 경로로부터 s1(GPT), s2(VITS), s2(CFM) 학습 설정을 생성한다.
각 스크립트의 _build_config_from_voice_dir()를 통합한다.
"""
from __future__ import annotations

import json
import os

import yaml
from loguru import logger

from tools.utils.download import ensure_file
from tools.training.data_helpers import merge_partitioned_files

_HF_REPO = "lj1995/GPT-SoVITS"
_DEFAULT_S1_TEMPLATE = "GPT_SoVITS/configs/s1longer-v2.yaml"

# ---------------------------------------------------------------------------
# Pretrained 모델 레지스트리
# ---------------------------------------------------------------------------

# s1 (GPT): (local_path, hf_filename, hf_subfolder)
PRETRAINED_S1 = {
    "v2": (
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "gsv-v2final-pretrained",
    ),
    "v3": ("GPT_SoVITS/pretrained_models/s1v3.ckpt", "s1v3.ckpt", None),
    "v4": ("GPT_SoVITS/pretrained_models/s1v3.ckpt", "s1v3.ckpt", None),
    "v2Pro": ("GPT_SoVITS/pretrained_models/s1v3.ckpt", "s1v3.ckpt", None),
    "v2ProPlus": ("GPT_SoVITS/pretrained_models/s1v3.ckpt", "s1v3.ckpt", None),
}

# s2 VITS (v2/v2Pro/v2ProPlus): (config_json, pretrained_s2G, hf_filename, hf_subfolder)
S2_VITS_CONFIGS = {
    "v2": (
        "GPT_SoVITS/configs/s2.json",
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "s2G2333k.pth",
        "gsv-v2final-pretrained",
    ),
    "v2Pro": (
        "GPT_SoVITS/configs/s2v2Pro.json",
        "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        "s2Gv2Pro.pth",
        "v2Pro",
    ),
    "v2ProPlus": (
        "GPT_SoVITS/configs/s2v2ProPlus.json",
        "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "s2Gv2ProPlus.pth",
        "v2Pro",
    ),
}

# s2 VITS Discriminator
DEFAULT_PRETRAINED_S2D = (
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
)

# s2 CFM (v3/v4): (config_json, pretrained_s2G, hf_filename, hf_subfolder)
S2_CFM_CONFIGS = {
    "v3": (
        "GPT_SoVITS/configs/s2.json",
        "GPT_SoVITS/pretrained_models/s2Gv3.pth",
        "s2Gv3.pth",
        None,
    ),
    "v4": (
        "GPT_SoVITS/configs/s2.json",
        "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
        "s2Gv4.pth",
        "gsv-v4-pretrained",
    ),
}


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _derive_paths(voice_dir: str, version: str) -> tuple[str, str, str]:
    """voice_dir로부터 전처리/학습 결과 경로를 파생한다.

    Returns:
        (preprocessed_dir, trained_dir, exp_name)
    """
    preprocessed = os.path.join(voice_dir, "step2", version)
    trained = os.path.join(voice_dir, "step3", version)
    exp_name = os.path.basename(os.path.normpath(voice_dir))
    return preprocessed, trained, exp_name


# ---------------------------------------------------------------------------
# s1 (GPT) 설정 생성
# ---------------------------------------------------------------------------


def build_s1_config(
    voice_dir: str,
    version: str = "v2Pro",
    template: str = _DEFAULT_S1_TEMPLATE,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    save_every_n_epoch: int | None = None,
) -> str:
    """GPT(s1) 학습 YAML 설정을 생성한다.

    Returns:
        저장된 설정 파일 경로.
    """
    preprocessed, trained, exp_name = _derive_paths(voice_dir, version)
    merge_partitioned_files(preprocessed)

    with open(template, "r") as f:
        config = yaml.full_load(f)

    # 데이터 경로
    config["train_semantic_path"] = os.path.join(preprocessed, "name2semantic.tsv")
    config["train_phoneme_path"] = os.path.join(preprocessed, "name2text.txt")
    config["output_dir"] = os.path.join(trained, "01_gpt_logs")

    # Pretrained 모델
    s1_path, s1_filename, s1_subfolder = PRETRAINED_S1[version]
    ensure_file(s1_path, _HF_REPO, filename=s1_filename, subfolder=s1_subfolder)
    config["pretrained_s1"] = s1_path

    # 학습 설정
    config["train"]["half_weights_save_dir"] = os.path.join(trained, "02_gpt_weights")
    config["train"]["exp_name"] = exp_name
    config["train"]["epochs"] = 10
    config.setdefault("train", {})
    config["train"].setdefault("if_save_latest", True)
    config["train"].setdefault("if_save_every_weights", True)
    config["train"].setdefault("if_dpo", False)
    config["train"].setdefault("precision", "16-mixed")

    # CLI 오버라이드
    if epochs is not None:
        config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if save_every_n_epoch is not None:
        config["train"]["save_every_n_epoch"] = save_every_n_epoch

    # 저장
    config_path = os.path.join(trained, "s1_config.yaml")
    os.makedirs(trained, exist_ok=True)
    os.makedirs(config["train"]["half_weights_save_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("s1 설정 생성: {}", config_path)
    return config_path


# ---------------------------------------------------------------------------
# s2 VITS (v2/v2Pro/v2ProPlus) 설정 생성
# ---------------------------------------------------------------------------


def build_s2_vits_config(
    voice_dir: str,
    version: str = "v2Pro",
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    save_every_epoch: int | None = None,
) -> str:
    """SoVITS VITS(s2) 학습 JSON 설정을 생성한다.

    Returns:
        저장된 설정 파일 경로.
    """
    preprocessed, trained, exp_name = _derive_paths(voice_dir, version)
    merge_partitioned_files(preprocessed)

    if version not in S2_VITS_CONFIGS:
        supported = list(S2_VITS_CONFIGS.keys())
        raise ValueError(
            f"VITS 학습은 {supported}만 지원합니다. v3/v4는 CFM 학습을 사용하세요."
        )
    s2_config_path, s2g_path, s2g_filename, s2g_subfolder = S2_VITS_CONFIGS[version]

    with open(s2_config_path, "r") as f:
        config = json.loads(f.read())

    # 데이터·체크포인트 경로
    config["data"]["exp_dir"] = preprocessed
    config["s2_ckpt_dir"] = os.path.join(trained, "03_sovits_logs")
    config["name"] = exp_name
    config["version"] = version
    config["save_weight_dir"] = os.path.join(trained, "04_sovits_weights")
    config["model"]["version"] = version

    # Pretrained Generator
    ensure_file(s2g_path, _HF_REPO, filename=s2g_filename, subfolder=s2g_subfolder)
    config["train"]["pretrained_s2G"] = s2g_path

    # Pretrained Discriminator
    if os.path.exists(DEFAULT_PRETRAINED_S2D):
        config["train"]["pretrained_s2D"] = DEFAULT_PRETRAINED_S2D
    else:
        try:
            ensure_file(
                DEFAULT_PRETRAINED_S2D,
                _HF_REPO,
                filename="s2D2333k.pth",
                subfolder="gsv-v2final-pretrained",
            )
            config["train"]["pretrained_s2D"] = DEFAULT_PRETRAINED_S2D
        except Exception:
            config["train"]["pretrained_s2D"] = ""

    # 학습 설정
    config["train"]["epochs"] = 10
    config["train"].setdefault("save_every_epoch", 4)
    config["train"].setdefault("if_save_latest", 1)
    config["train"].setdefault("if_save_every_weights", True)
    config["train"].setdefault("gpu_numbers", "0")

    # CLI 오버라이드
    if epochs is not None:
        config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if save_every_epoch is not None:
        config["train"]["save_every_epoch"] = save_every_epoch

    # 저장
    config_path = os.path.join(trained, "s2_config.json")
    os.makedirs(trained, exist_ok=True)
    os.makedirs(config["s2_ckpt_dir"], exist_ok=True)
    os.makedirs(config["save_weight_dir"], exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("s2 VITS 설정 생성: {}", config_path)
    return config_path


# ---------------------------------------------------------------------------
# s2 CFM (v3/v4) 설정 생성
# ---------------------------------------------------------------------------


def build_s2_cfm_config(
    voice_dir: str,
    version: str = "v3",
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    save_every_epoch: int | None = None,
) -> str:
    """SoVITS CFM(s2 v3/v4) 학습 JSON 설정을 생성한다.

    Returns:
        저장된 설정 파일 경로.
    """
    preprocessed, trained, exp_name = _derive_paths(voice_dir, version)

    if version not in S2_CFM_CONFIGS:
        supported = list(S2_CFM_CONFIGS.keys())
        raise ValueError(f"CFM 학습은 {supported}만 지원합니다.")
    s2_config_path, s2g_path, s2g_filename, s2g_subfolder = S2_CFM_CONFIGS[version]

    with open(s2_config_path, "r") as f:
        config = json.loads(f.read())

    # 데이터·체크포인트 경로
    config["data"]["exp_dir"] = preprocessed
    config["s2_ckpt_dir"] = os.path.join(trained, "03_sovits_logs")
    config["name"] = exp_name
    config["version"] = version
    config["save_weight_dir"] = os.path.join(trained, "04_sovits_weights")
    config["model"]["version"] = version

    # Pretrained Generator
    ensure_file(s2g_path, _HF_REPO, filename=s2g_filename, subfolder=s2g_subfolder)
    config["train"]["pretrained_s2G"] = s2g_path

    # 학습 설정 — 템플릿 값을 v3/v4에 맞게 덮어쓴다
    config["train"]["epochs"] = 10
    config["train"]["batch_size"] = 1
    config["train"]["grad_ckpt"] = True
    config["train"].setdefault("save_every_epoch", 4)
    config["train"].setdefault("if_save_latest", 1)
    config["train"].setdefault("if_save_every_weights", True)
    config["train"].setdefault("gpu_numbers", "0")

    # CLI 오버라이드
    if epochs is not None:
        config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if save_every_epoch is not None:
        config["train"]["save_every_epoch"] = save_every_epoch

    # 저장
    config_path = os.path.join(trained, "s2_config.json")
    os.makedirs(trained, exist_ok=True)
    os.makedirs(config["s2_ckpt_dir"], exist_ok=True)
    os.makedirs(config["save_weight_dir"], exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("s2 CFM 설정 생성: {}", config_path)
    return config_path
