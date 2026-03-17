"""디바이스 설정 + 텐서 이동 헬퍼."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """사용 가능한 최적 디바이스를 반환한다."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_batch_to_device(
    batch: tuple | list,
    device: torch.device,
) -> tuple:
    """배치 내 텐서를 일괄 이동한다."""
    return tuple(
        t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
        for t in batch
    )


def configure_torch_backends() -> None:
    """cuDNN / TF32 등 학습 백엔드를 설정한다."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
