# -*- coding: utf-8 -*-
"""GPT (AR) 모델 학습 (Stage 1).

--voice-dir 로 캐릭터 음성 폴더를 지정하면
전처리 결과(step2)에서 데이터 경로를 자동 파생한다.

출력:
  - {voice-dir}/step3/{version}/01_gpt_logs/       (체크포인트 + 텐서보드)
  - {voice-dir}/step3/{version}/02_gpt_weights/    (half-precision 가중치)
"""
# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# -- 경로 부트스트랩 --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# ---------------------------------------------------------------

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

import pathlib

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils import get_newest_ckpt
from AR.utils.io import load_yaml_config
from loguru import logger as log
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tools.training.checkpoint import save_gpt_weight_checkpoint
from tools.training.config_builder import PRETRAINED_S1, build_s1_config

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Checkpoint 콜백
# ---------------------------------------------------------------------------


class _GPTModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.if_save_latest:
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except OSError as e:
                            log.warning("이전 체크포인트 삭제 실패: {} ({})", name, e)
                if self.if_save_every_weights:
                    state_dict = trainer.strategy._lightning_module.state_dict()
                    save_gpt_weight_checkpoint(
                        state_dict,
                        self.config,
                        trainer.current_epoch + 1,
                        self.half_weights_save_dir,
                        self.exp_name,
                    )
            self._save_last_checkpoint(trainer, monitor_candidates)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def train(config_path: str) -> None:
    os.environ["hz"] = "25hz"
    config = load_yaml_config(config_path)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback = _GPTModelCheckpoint(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["USE_LIBUV"] = "0"
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        limit_val_batches=0,
        devices=1,
        benchmark=False,
        fast_dev_run=False,
        strategy="auto",
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,
    )

    model = Text2SemanticLightningModule(config, output_dir)
    data_module = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    log.info("ckpt_path: {}", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT (AR) 모델 학습")
    parser.add_argument("--voice-dir", default=None, help="캐릭터 음성 폴더 (예: data/voice/lunabi)")
    parser.add_argument(
        "--version", default="v2Pro",
        choices=list(PRETRAINED_S1.keys()),
        help="모델 버전 (기본: v2Pro). step3/{version}/ 하위에 가중치를 저장한다.",
    )
    parser.add_argument(
        "-c", "--config-file", default=None,
        help="YAML 설정 파일 경로 (--voice-dir 미지정 시 사용)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--save-every-n-epoch", type=int, default=None, help="체크포인트 저장 주기")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.voice_dir:
        config_path = build_s1_config(
            args.voice_dir,
            version=args.version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_every_n_epoch=args.save_every_n_epoch,
        )
    elif args.config_file:
        config_path = args.config_file
    else:
        log.error("--voice-dir 또는 -c/--config-file 중 하나를 지정하세요.")
        sys.exit(1)

    train(config_path)
