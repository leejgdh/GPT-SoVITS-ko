"""학습 루프 공통 헬퍼.

학습 스크립트에서 반복되는 backward/step 패턴과
전역 변수(global_step)를 대체하는 TrainingState를 제공한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.cuda.amp import GradScaler


@dataclass
class TrainingState:
    """학습 상태 추적.

    전역 변수 global_step / epoch를 대체한다.
    """

    global_step: int = 0
    epoch: int = 1


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    parameters,
) -> float:
    """공통 backward -> grad clip -> optimizer step 패턴.

    scaler.update()는 호출하지 않는다.
    GAN 학습에서 D/G 스텝 이후 한 번만 호출해야 하므로
    호출부에서 직접 scaler.update()를 수행한다.

    Args:
        loss: 스케일링 전 손실.
        optimizer: 옵티마이저.
        scaler: GradScaler (fp16_run=False이면 no-op).
        parameters: 그래디언트 클리핑 대상 파라미터.

    Returns:
        그래디언트 노름.
    """
    from module.commons import clip_grad_value_

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_value_(parameters, None)
    scaler.step(optimizer)
    return grad_norm
