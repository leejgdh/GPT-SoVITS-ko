# -*- coding: utf-8 -*-
"""SoVITS (VITS) 모델 학습 (Stage 2).

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
import json
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
from module import commons
from module.data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tools.training.checkpoint import (
    resume_or_load_pretrained,
    save_epoch_checkpoint,
    save_latest_checkpoint,
    save_weight_checkpoint,
)
from tools.training.config_builder import build_s2_vits_config
from tools.training.data_helpers import SingleGPUBucketSampler
from tools.training.device import (
    configure_torch_backends,
    get_device,
    move_batch_to_device,
)
from tools.training.logging_utils import (
    plot_spectrogram_to_numpy,
    setup_training_logger,
    summarize,
)
from tools.training.loop import TrainingState, backward_and_step

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

configure_torch_backends()

# 버킷 경계값 (시퀀스 길이 기준)
_BUCKET_BOUNDARIES = [
    32, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
]


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------


def train(hps):
    """VITS (v1/v2/v2Pro/v2ProPlus) 학습."""
    device = get_device()

    setup_training_logger(hps.data.exp_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
    torch.manual_seed(hps.train.seed)

    # -- 데이터 로더 --
    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.model.version)
    train_sampler = SingleGPUBucketSampler(
        train_dataset,
        hps.train.batch_size,
        _BUCKET_BOUNDARIES,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate(version=hps.model.version)
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

    # -- 모델 생성 --
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(
        hps.model.use_spectral_norm, version=hps.model.version,
    ).to(device)

    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            logger.debug("{} not requires_grad", name)

    # -- 옵티마이저 (text_embedding / encoder_text / mrte에 별도 LR 적용) --
    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )

    optim_g = torch.optim.AdamW(
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {"params": net_g.enc_p.text_embedding.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
            {"params": net_g.enc_p.encoder_text.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
            {"params": net_g.enc_p.mrte.parameters(), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
        ],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # -- 체크포인트 복원 또는 pretrained 로드 --
    start_epoch, global_step = resume_or_load_pretrained(
        ckpt_dir=hps.s2_ckpt_dir,
        model=net_g,
        optimizer=optim_g,
        pretrained_path=hps.train.pretrained_s2G if hps.train.pretrained_s2G else None,
        train_loader_len=len(train_loader),
        ckpt_pattern="G_*.pth",
        discriminator=net_d,
        optim_d=optim_d,
        pretrained_d_path=hps.train.pretrained_s2D if hps.train.pretrained_s2D else None,
    )
    state = TrainingState(global_step=global_step, epoch=start_epoch)

    # -- 학습률 스케줄러 --
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(start_epoch):
        scheduler_g.step()
        scheduler_d.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    logger.info("start training from epoch {}", start_epoch)
    for epoch in range(start_epoch, hps.train.epochs + 1):
        state.epoch = epoch
        _train_one_epoch(
            epoch, hps, net_g, net_d, optim_g, optim_d,
            scaler, train_loader, writer, state, device,
        )
        scheduler_g.step()
        scheduler_d.step()
    logger.info("training done")


def _train_one_epoch(epoch, hps, net_g, net_d, optim_g, optim_d, scaler, train_loader, writer, state, device):
    """1 에포크 학습 + 체크포인트 저장."""
    train_loader.batch_sampler.set_epoch(epoch)
    is_pro = hps.model.version in {"v2Pro", "v2ProPlus"}

    net_g.train()
    net_d.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # -- 배치 텐서를 디바이스로 이동 --
        data = move_batch_to_device(data, device)

        if is_pro:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
        else:
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data

        ssl = ssl.detach()  # ssl은 그래디언트 불필요

        # ===== Discriminator 스텝 =====
        with autocast(enabled=hps.train.fp16_run):
            if is_pro:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths, sv_emb,
                )
            else:
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lengths, text, text_lengths,
                )

            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        grad_norm_d = backward_and_step(loss_disc_all, optim_d, scaler, net_d.parameters())

        # ===== Generator 스텝 =====
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

        grad_norm_g = backward_and_step(loss_gen_all, optim_g, scaler, net_g.parameters())
        scaler.update()

        # -- 텐서보드 로깅 --
        if state.global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
            logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
            logger.info([x.item() for x in losses] + [state.global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all,
                "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update({
                "loss/g/fm": loss_fm, "loss/g/mel": loss_mel,
                "loss/g/kl_ssl": kl_ssl, "loss/g/kl": loss_kl,
            })
            image_dict = None
            try:
                image_dict = {
                    "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/stats_ssl": plot_spectrogram_to_numpy(stats_ssl[0].data.cpu().numpy()),
                }
            except Exception:
                pass
            if image_dict:
                summarize(writer=writer, global_step=state.global_step, images=image_dict, scalars=scalar_dict)
            else:
                summarize(writer=writer, global_step=state.global_step, scalars=scalar_dict)

        state.global_step += 1

    # -- 에포크 종료 시 체크포인트 저장 --
    if epoch % hps.train.save_every_epoch == 0:
        if hps.train.if_save_latest:
            save_latest_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, hps.s2_ckpt_dir, prefix="G")
            save_latest_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, hps.s2_ckpt_dir, prefix="D")
        else:
            save_epoch_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, state.global_step, hps.s2_ckpt_dir, prefix="G")
            save_epoch_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, state.global_step, hps.s2_ckpt_dir, prefix="D")

        if hps.train.if_save_every_weights:
            ckpt = net_g.state_dict()
            model_version = hps.model.version if hps.model.version in {"v2Pro", "v2ProPlus"} else None
            logger.info(
                "saving ckpt %s_e%s:%s" % (
                    hps.name, epoch,
                    save_weight_checkpoint(
                        ckpt, hps.name + "_e%s_s%s" % (epoch, state.global_step),
                        epoch, state.global_step, hps,
                        model_version=model_version,
                    ),
                )
            )

    logger.info("====> Epoch: {}".format(epoch))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoVITS (VITS) 모델 학습")
    parser.add_argument("--voice-dir", default=None, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument(
        "-c", "--config", default=None,
        help="JSON 설정 파일 경로 (--voice-dir 미지정 시 사용)",
    )
    parser.add_argument("--version", default="v2Pro", help="모델 버전 (기본: v2Pro)")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--save-every-epoch", type=int, default=None, help="체크포인트 저장 주기")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.voice_dir:
        config_path = build_s2_vits_config(
            args.voice_dir,
            version=args.version or "v2Pro",
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_every_epoch=args.save_every_epoch,
        )
    elif args.config:
        config_path = args.config
    else:
        logger.error("--voice-dir 또는 -c/--config 중 하나를 지정하세요.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.loads(f.read())
    hps = utils.HParams(**config)

    gpu_numbers = getattr(hps.train, "gpu_numbers", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")

    os.makedirs(hps.s2_ckpt_dir, exist_ok=True)
    config_save_path = os.path.join(hps.s2_ckpt_dir, "config.json")
    with open(config_save_path, "w") as f:
        f.write(json.dumps(config, indent=2, ensure_ascii=False))

    train(hps)
