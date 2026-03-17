# -*- coding: utf-8 -*-
"""SoVITS CFM (v3/v4) 모델 학습 — 싱글 GPU.

--voice-dir 로 캐릭터 음성 폴더를 지정하면
전처리 결과(step2)에서 데이터 경로를 자동 파생한다.

출력:
  - {voice-dir}/step3/{version}/03_sovits_logs/       (체크포인트 + 텐서보드)
  - {voice-dir}/step3/{version}/04_sovits_weights/    (half-precision 가중치)
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import argparse
import logging
import os
import sys

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# ---------------------------------------------------------------

import torch
import utils
from loguru import logger
from module.data_utils import (
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerLoaderV4,
)
from module.models import SynthesizerTrnV3 as SynthesizerTrn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tools.training.checkpoint import (
    resume_or_load_pretrained,
    save_epoch_checkpoint,
    save_latest_checkpoint,
    save_weight_checkpoint,
)
from tools.training.config_builder import S2_CFM_CONFIGS, build_s2_cfm_config
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
# 학습
# ---------------------------------------------------------------------------


def train(hps) -> None:
    """CFM (v3/v4) 학습 메인 함수 — 싱글 GPU."""
    device = get_device()
    setup_training_logger(hps.data.exp_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)

    torch.manual_seed(hps.train.seed)

    # v3/v4 버전에 따라 데이터 로더·콜레이트 선택
    # v3: 24kHz mel (n_fft=1024, hop=256)
    # v4: 32kHz mel (n_fft=1280, hop=320)
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

    # 모델 생성
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # 체크포인트 복원 또는 pretrained 로드
    state = TrainingState()
    start_epoch, global_step = resume_or_load_pretrained(
        hps.s2_ckpt_dir,
        net_g,
        optim_g,
        pretrained_path=hps.train.pretrained_s2G,
        train_loader_len=len(train_loader),
    )
    state.epoch = start_epoch
    state.global_step = global_step

    # 스케줄러: 복원된 에포크까지 step 진행
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1,
    )
    for _ in range(start_epoch):
        scheduler_g.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    logger.info("에포크 {}부터 학습 시작", start_epoch)
    for epoch in range(start_epoch, hps.train.epochs + 1):
        state.epoch = epoch
        _train_one_epoch(hps, net_g, optim_g, scaler, train_loader, state, device, writer)
        scheduler_g.step()

    logger.info("학습 완료")


def _train_one_epoch(
    hps,
    net_g: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    state: TrainingState,
    device: torch.device,
    writer: SummaryWriter,
) -> None:
    """1 에포크 학습 + 체크포인트 저장."""
    epoch = state.epoch
    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths = (
            move_batch_to_device(batch, device)
        )
        ssl.requires_grad = False

        with autocast(enabled=hps.train.fp16_run):
            cfm_loss = net_g(
                ssl, spec, mel, ssl_lengths, spec_lengths,
                text, text_lengths, mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss

        grad_norm_g = backward_and_step(loss_gen_all, optim_g, scaler, net_g.parameters())
        scaler.update()

        # 로깅
        if state.global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(
                "Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader))
            )
            logger.info([cfm_loss.item(), state.global_step, lr])
            summarize(
                writer=writer,
                global_step=state.global_step,
                scalars={
                    "loss/g/total": loss_gen_all,
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g,
                },
            )

        state.global_step += 1

    # 에포크 체크포인트 저장
    if epoch % hps.train.save_every_epoch == 0:
        if hps.train.if_save_latest:
            save_latest_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                hps.s2_ckpt_dir,
            )
        else:
            save_epoch_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                state.global_step, hps.s2_ckpt_dir,
            )

        # half-precision 가중치 저장
        if hps.train.if_save_every_weights:
            ckpt = net_g.state_dict()
            result = save_weight_checkpoint(
                ckpt,
                hps.name + "_e%s_s%s" % (epoch, state.global_step),
                epoch,
                state.global_step,
                hps,
                model_version=hps.model.version,
            )
            logger.info("saving ckpt %s_e%s:%s" % (hps.name, epoch, result))

    logger.info("====> Epoch: {}".format(epoch))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoVITS CFM (v3/v4) 모델 학습 — 싱글 GPU")
    parser.add_argument("--voice-dir", default=None, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument(
        "--version", default="v3",
        choices=list(S2_CFM_CONFIGS.keys()),
        help="모델 버전 (기본: v3). step3/{version}/ 하위에 가중치를 저장한다.",
    )
    parser.add_argument(
        "-c", "--config-file", default=None,
        help="JSON 설정 파일 경로 (--voice-dir 미지정 시 사용)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--save-every-epoch", type=int, default=None, help="체크포인트 저장 주기")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.voice_dir:
        config_path = build_s2_cfm_config(
            args.voice_dir,
            args.version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_every_epoch=args.save_every_epoch,
        )
    elif args.config_file:
        config_path = args.config_file
    else:
        logger.error("--voice-dir 또는 -c/--config-file 중 하나를 지정하세요.")
        sys.exit(1)

    hps = utils.get_hparams_from_file(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")

    train(hps)
