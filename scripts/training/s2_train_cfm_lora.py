# -*- coding: utf-8 -*-
"""SoVITS v3/v4 CFM LoRA 학습 스크립트.

CFM(Conditional Flow Matching) 모듈에 LoRA(Low-Rank Adaptation)를 적용하여
소량의 파라미터만 학습한다. DDP 없이 싱글 GPU 전용으로 동작한다.

사용법:
  python s2_train_cfm_lora.py -c /path/to/config.json
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import logging
import os
import sys
from collections import OrderedDict

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# ---------------------------------------------------------------

import utils

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")

import torch
from loguru import logger
from module.data_utils import (
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerLoaderV4,
)
from module.models import SynthesizerTrnV3 as SynthesizerTrn
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tools.training.checkpoint import (
    latest_checkpoint_path,
    load_training_checkpoint,
    save_epoch_checkpoint,
    save_latest_checkpoint,
    save_weight_checkpoint,
)
from tools.training.data_helpers import SingleGPUBucketSampler
from tools.training.device import (
    configure_torch_backends,
    get_device,
    move_batch_to_device,
)
from tools.training.logging_utils import setup_training_logger, summarize
from tools.training.loop import TrainingState, backward_and_step

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

configure_torch_backends()


# ---------------------------------------------------------------------------
# 모델 생성
# ---------------------------------------------------------------------------


def _build_model(hps) -> SynthesizerTrn:
    """SynthesizerTrn 모델을 생성한다."""
    return SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )


def _build_optimizer(net_g: torch.nn.Module, hps) -> torch.optim.AdamW:
    """학습 가능한 파라미터만 대상으로 옵티마이저를 생성한다."""
    # 모든 레이어 동일 LR
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------


def _train_one_epoch(
    epoch: int,
    hps,
    net_g: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    state: TrainingState,
    device: torch.device,
    no_grad_names: set[str],
    save_root: str,
    lora_rank: int,
    writer: SummaryWriter,
) -> None:
    """1 에포크 학습 + 체크포인트 저장."""
    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths = (
            move_batch_to_device(batch, device)
        )
        ssl.requires_grad = False

        with autocast(enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss

        grad_norm_g = backward_and_step(loss_gen_all, optim_g, scaler, net_g.parameters())
        scaler.update()

        if state.global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [cfm_loss]
            logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
            logger.info([x.item() for x in losses] + [state.global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "learning_rate": lr,
                "grad_norm_g": grad_norm_g,
            }
            summarize(writer=writer, global_step=state.global_step, scalars=scalar_dict)

        state.global_step += 1

    # -- 에포크 종료: 체크포인트 저장 --
    if epoch % hps.train.save_every_epoch == 0:
        if hps.train.if_save_latest:
            save_latest_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                save_root,
            )
        else:
            save_epoch_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                state.global_step, save_root,
            )

        if hps.train.if_save_every_weights:
            ckpt = net_g.state_dict()
            # LoRA: 학습 가능 파라미터만 저장
            sim_ckpt = OrderedDict()
            for key in ckpt:
                if key not in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    save_weight_checkpoint(
                        sim_ckpt,
                        hps.name + "_e%s_s%s_l%s" % (epoch, state.global_step, lora_rank),
                        epoch,
                        state.global_step,
                        hps,
                        model_version=hps.model.version,
                        lora_rank=lora_rank,
                    ),
                )
            )

    logger.info("====> Epoch: {}".format(epoch))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def train() -> None:
    """CFM LoRA (v3/v4) 학습."""
    device = get_device()
    torch.manual_seed(hps.train.seed)

    setup_training_logger(hps.data.exp_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)

    # -- v3/v4 데이터 로더 선택 --
    if hps.model.version == "v3":
        TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3
        TextAudioSpeakerCollate = TextAudioSpeakerCollateV3
    else:
        TextAudioSpeakerLoader = TextAudioSpeakerLoaderV4
        TextAudioSpeakerCollate = TextAudioSpeakerCollateV4

    train_dataset = TextAudioSpeakerLoader(hps.data)
    train_sampler = SingleGPUBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=3,
    )

    # -- LoRA 설정 --
    save_root = "%s/logs_s2_%s_lora_%s" % (hps.data.exp_dir, hps.model.version, hps.train.lora_rank)
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=True,
    )

    # -- 모델·옵티마이저 구축 + 체크포인트 복원 --
    state = TrainingState()

    try:
        # 체크포인트 자동 복원 시도
        net_g = _build_model(hps)
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = net_g.to(device)
        optim_g = _build_optimizer(net_g, hps)
        _, _, _, start_epoch = load_training_checkpoint(
            latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )
        start_epoch += 1
        state.global_step = (start_epoch - 1) * len(train_loader)
    except Exception:
        # 복원 실패 시 pretrained 로드
        start_epoch = 1
        state.global_step = 0
        net_g = _build_model(hps)
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G is not None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            logger.info(
                "loaded pretrained {}: {}",
                hps.train.pretrained_s2G,
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)["weight"],
                    strict=False,
                ),
            )
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = net_g.to(device)
        optim_g = _build_optimizer(net_g, hps)

    state.epoch = start_epoch

    # -- 학습 불가 파라미터 이름 수집 (LoRA 저장용) --
    no_grad_names: set[str] = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(start_epoch):
        scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    logger.info("start training from epoch {}", start_epoch)
    for epoch in range(start_epoch, hps.train.epochs + 1):
        state.epoch = epoch
        _train_one_epoch(
            epoch,
            hps,
            net_g,
            optim_g,
            scaler,
            train_loader,
            state,
            device,
            no_grad_names,
            save_root,
            lora_rank,
            writer,
        )
        scheduler_g.step()
    logger.info("training done")


if __name__ == "__main__":
    train()
