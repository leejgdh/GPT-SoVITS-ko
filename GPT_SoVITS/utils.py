"""설정(HParams) + 유틸리티.

학습·추론에서 공통으로 사용하는 HParams 클래스와
설정 파일 로드 함수를 제공한다.

학습 전용 유틸리티는 tools/training/ 패키지를 참조.
"""
import argparse
import json
import logging
import os
import sys

logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HParams
# ---------------------------------------------------------------------------


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


# ---------------------------------------------------------------------------
# 설정 파일 로드
# ---------------------------------------------------------------------------


def get_hparams_from_file(config_path: str) -> HParams:
    """JSON 설정 파일을 HParams 객체로 로드한다."""
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    return HParams(**config)


def get_hparams(init: bool = True, stage: int = 1) -> HParams:
    """CLI 인자에서 설정 파일을 읽어 HParams를 반환한다.

    주로 s2_train_cfm_lora.py에서 사용.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str,
        default="./configs/s2.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-p", "--pretrain", type=str, required=False, default=None, help="pretrain dir")
    parser.add_argument("-rs", "--resume_step", type=int, required=False, default=None, help="resume step")

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.pretrain = args.pretrain
    hparams.resume_step = args.resume_step

    if stage == 1:
        model_dir = hparams.s1_ckpt_dir
    else:
        model_dir = hparams.s2_ckpt_dir
    config_save_path = os.path.join(model_dir, "config.json")

    os.makedirs(model_dir, exist_ok=True)
    with open(config_save_path, "w") as f:
        f.write(data)
    return hparams
