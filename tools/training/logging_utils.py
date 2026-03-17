"""학습 로깅 유틸리티.

텐서보드 SummaryWriter 헬퍼 + 로그 파일 설정을 제공한다.
프로젝트 공통 로거(loguru)를 사용한다.
"""
from __future__ import annotations

import os

import numpy as np
from loguru import logger


def setup_training_logger(log_dir: str, filename: str = "train.log") -> None:
    """학습 로그 파일 싱크를 추가한다.

    loguru의 글로벌 logger에 파일 싱크를 추가하므로
    호출 후 ``from loguru import logger`` 로 사용하면 된다.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, filename),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss}\t{name}\t{level}\t{message}",
        rotation=None,
    )


def summarize(
    writer,
    global_step: int,
    scalars: dict | None = None,
    histograms: dict | None = None,
    images: dict | None = None,
    audios: dict | None = None,
    audio_sampling_rate: int = 22050,
) -> None:
    """텐서보드에 학습 메트릭을 기록한다."""
    if scalars:
        for k, v in scalars.items():
            writer.add_scalar(k, v, global_step)
    if histograms:
        for k, v in histograms.items():
            writer.add_histogram(k, v, global_step)
    if images:
        for k, v in images.items():
            writer.add_image(k, v, global_step, dataformats="HWC")
    if audios:
        for k, v in audios.items():
            writer.add_audio(k, v, global_step, audio_sampling_rate)


def plot_spectrogram_to_numpy(spectrogram) -> np.ndarray:
    """스펙트로그램을 텐서보드 이미지용 numpy 배열로 변환한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
