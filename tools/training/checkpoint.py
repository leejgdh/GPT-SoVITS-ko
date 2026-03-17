"""체크포인트 저장/로드 통합 모듈."""
from __future__ import annotations

import glob
import os
import shutil
import traceback
from collections import OrderedDict
from io import BytesIO
from time import time

import torch
from loguru import logger

# 모델 버전 → 파일 헤더 바이트 매핑 (process_ckpt.head2version과 대응)
_VERSION_TO_BYTE = {
    "v3": b"02",
    "v4": b"07",
    "v2Pro": b"05",
    "v2ProPlus": b"06",
}
_LORA_VERSION_TO_BYTE = {
    "v3": b"03",
    "v4": b"04",
}


def _save_via_temp(data: object, path: str) -> None:
    """torch.save가 비ASCII 경로에서 실패하는 문제를 우회한다."""
    dir_ = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % time()
    torch.save(data, tmp_path)
    shutil.move(tmp_path, os.path.join(dir_, name))


def _save_with_version_header(data: object, path: str, version_byte: bytes) -> None:
    """버전 헤더를 포함하여 저장한다 (v3/v4/v2Pro/v2ProPlus)."""
    bio = BytesIO()
    torch.save(data, bio)
    bio.seek(0)
    raw = bio.getvalue()
    raw = version_byte + raw[2:]
    with open(path, "wb") as f:
        f.write(raw)


# ---------------------------------------------------------------------------
# 학습 체크포인트 (optimizer state 포함)
# ---------------------------------------------------------------------------


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epoch: int,
    path: str,
) -> None:
    """학습 체크포인트를 저장한다 (모델 + optimizer + 에포크)."""
    logger.info("Saving checkpoint at epoch {} to {}", epoch, path)
    state_dict = model.state_dict()
    _save_via_temp(
        {
            "model": state_dict,
            "iteration": epoch,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        path,
    )


def save_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epoch: int,
    ckpt_dir: str,
    prefix: str = "G",
) -> None:
    """최신 체크포인트를 저장한다 (이전 latest 덮어쓰기)."""
    path = os.path.join(ckpt_dir, f"{prefix}_latest.pth")
    save_training_checkpoint(model, optimizer, learning_rate, epoch, path)


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epoch: int,
    global_step: int,
    ckpt_dir: str,
    prefix: str = "G",
) -> None:
    """에포크별 체크포인트를 저장한다."""
    path = os.path.join(ckpt_dir, f"{prefix}_{global_step}.pth")
    save_training_checkpoint(model, optimizer, learning_rate, epoch, path)


# ---------------------------------------------------------------------------
# 가중치 체크포인트 (half-precision, 추론/파인튜닝용)
# ---------------------------------------------------------------------------


def save_weight_checkpoint(
    state_dict: dict,
    name: str,
    epoch: int,
    steps: int,
    hps,
    model_version: str | None = None,
    lora_rank: int | None = None,
) -> str:
    """Half-precision 가중치를 저장한다 (추론/파인튜닝용)."""
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in state_dict.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = state_dict[key].half()
        opt["config"] = hps
        opt["info"] = "%sepoch_%siteration" % (epoch, steps)
        if lora_rank:
            opt["lora_rank"] = lora_rank

        save_path = os.path.join(hps.save_weight_dir, f"{name}.pth")
        if lora_rank and model_version in _LORA_VERSION_TO_BYTE:
            _save_with_version_header(opt, save_path, _LORA_VERSION_TO_BYTE[model_version])
        elif model_version is not None and model_version in _VERSION_TO_BYTE:
            _save_with_version_header(opt, save_path, _VERSION_TO_BYTE[model_version])
        else:
            _save_via_temp(opt, save_path)
        return "Success."
    except Exception:
        return traceback.format_exc()


def save_gpt_weight_checkpoint(
    state_dict: dict,
    config: dict,
    epoch: int,
    save_dir: str,
    exp_name: str,
) -> None:
    """GPT(s1) 모델 가중치를 half-precision으로 저장한다."""
    opt = OrderedDict()
    opt["weight"] = OrderedDict()
    for key in state_dict:
        opt["weight"][key] = state_dict[key].half()
    opt["config"] = config
    opt["info"] = "GPT-e%s" % epoch
    path = os.path.join(save_dir, f"{exp_name}-e{epoch}.ckpt")
    _save_via_temp(opt, path)


# ---------------------------------------------------------------------------
# 체크포인트 로드
# ---------------------------------------------------------------------------


def load_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    skip_optimizer: bool = False,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    """학습 체크포인트를 로드한다.

    Returns:
        (model, optimizer, learning_rate, epoch)
    """
    assert os.path.isfile(path), f"체크포인트를 찾을 수 없습니다: {path}"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint["iteration"]
    learning_rate = checkpoint["learning_rate"]

    if optimizer is not None and not skip_optimizer and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    saved_state = checkpoint["model"]
    current_state = model.state_dict()
    new_state = {}
    for k, v in current_state.items():
        try:
            new_state[k] = saved_state[k]
            assert saved_state[k].shape == v.shape, (saved_state[k].shape, v.shape)
        except Exception:
            traceback.print_exc()
            logger.error("error, {} is not in the checkpoint", k)
            new_state[k] = v

    model.load_state_dict(new_state)
    logger.info("Loaded checkpoint '{}' (epoch {})", path, epoch)
    return model, optimizer, learning_rate, epoch


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> str:
    """디렉토리에서 최신 체크포인트 경로를 반환한다."""
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return f_list[-1]


def _load_pretrained_filtered(
    model: torch.nn.Module, state: dict
) -> str:
    """Shape이 일치하는 키만 로드한다 (PyTorch 2.6+ shape mismatch 대응)."""
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    skipped = set(state.keys()) - set(filtered.keys())
    result = model.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning("shape mismatch로 스킵된 키: {}", skipped)
    return str(result)


def resume_or_load_pretrained(
    ckpt_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pretrained_path: str | None = None,
    train_loader_len: int = 1,
    ckpt_pattern: str = "G_*.pth",
    discriminator: torch.nn.Module | None = None,
    optim_d: torch.optim.Optimizer | None = None,
    pretrained_d_path: str | None = None,
) -> tuple[int, int]:
    """체크포인트 복원 또는 pretrained 로드.

    Returns:
        (start_epoch, global_step)
    """
    try:
        _, _, _, epoch = load_training_checkpoint(
            latest_checkpoint_path(ckpt_dir, ckpt_pattern),
            model, optimizer,
        )
        if discriminator is not None and optim_d is not None:
            load_training_checkpoint(
                latest_checkpoint_path(ckpt_dir, "D_*.pth"),
                discriminator, optim_d,
            )
        start_epoch = epoch + 1
        global_step = (start_epoch - 1) * train_loader_len
        logger.info("Resumed from epoch {}", epoch)
    except Exception:
        start_epoch = 1
        global_step = 0
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)["weight"]
            load_result = _load_pretrained_filtered(model, state)
            logger.info("loaded pretrained {}: {}", pretrained_path, load_result)
        if discriminator is not None and pretrained_d_path and os.path.exists(pretrained_d_path):
            state_d = torch.load(pretrained_d_path, map_location="cpu", weights_only=False)["weight"]
            load_result_d = _load_pretrained_filtered(discriminator, state_d)
            logger.info("loaded pretrained D {}: {}", pretrained_d_path, load_result_d)

    return start_epoch, global_step
