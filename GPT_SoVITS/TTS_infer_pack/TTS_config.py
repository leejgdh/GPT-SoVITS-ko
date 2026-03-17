"""TTS 설정 관리.

모든 버전(v1~v4, v2Pro, v2ProPlus)의 기본 설정을 정의하고,
YAML 파일과의 직렬화/역직렬화를 담당한다.
"""
from __future__ import annotations

import os
from copy import deepcopy

import torch
import yaml
from loguru import logger


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class TTS_Config:
    default_configs = {
        "v1": {
            "device": "cpu",
            "is_half": False,
            "version": "v1",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
        "v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
        "v3": {
            "device": "cpu",
            "is_half": False,
            "version": "v3",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
        "v4": {
            "device": "cpu",
            "is_half": False,
            "version": "v4",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
        "v2Pro": {
            "device": "cpu",
            "is_half": False,
            "version": "v2Pro",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
        "v2ProPlus": {
            "device": "cpu",
            "is_half": False,
            "version": "v2ProPlus",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
            "hubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        },
    }
    configs: dict = None
    v1_languages: list = ["auto", "en", "ja", "ko"]
    v2_languages: list = ["auto", "en", "ja", "ko", "all_ja", "all_ko"]
    languages: list = v2_languages
    mute_tokens: dict = {
        "v1": 486,
        "v2": 486,
        "v2Pro": 486,
        "v2ProPlus": 486,
        "v3": 486,
        "v4": 486,
    }
    mute_emb_sim_matrix: torch.Tensor = None

    def __init__(self, configs: dict = None, config_save_path: str = None):
        self.configs_path: str | None = config_save_path

        if configs in ["", None]:
            configs = deepcopy(self.default_configs)

        assert isinstance(configs, dict)
        configs_ = deepcopy(self.default_configs)
        configs_.update(configs)
        self.configs: dict = configs_.get("custom", configs_["v2Pro"])
        self.default_configs = deepcopy(configs_)

        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            logger.warning("CUDA 사용 불가, CPU로 설정합니다")
            self.device = torch.device("cpu")

        self.is_half = self.configs.get("is_half", False)
        if str(self.device) == "cpu" and self.is_half:
            logger.warning("CPU에서는 half precision을 지원하지 않아 비활성화합니다")
            self.is_half = False

        version = self.configs.get("version", None)
        self.version = version
        assert self.version in ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"], "Invalid version!"
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.hubert_base_path = self.configs.get("hubert_base_path", None)
        self.languages = self.v1_languages if self.version == "v1" else self.v2_languages

        self.use_vocoder: bool = False

        if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
            self.t2s_weights_path = self.default_configs[version]["t2s_weights_path"]
            logger.warning("기본 t2s_weights_path로 대체: {}", self.t2s_weights_path)
        if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
            self.vits_weights_path = self.default_configs[version]["vits_weights_path"]
            logger.warning("기본 vits_weights_path로 대체: {}", self.vits_weights_path)
        if (self.hubert_base_path in [None, ""]) or (not os.path.exists(self.hubert_base_path)):
            self.hubert_base_path = self.default_configs[version]["hubert_base_path"]
            logger.warning("기본 hubert_base_path로 대체: {}", self.hubert_base_path)
        self.update_configs()

        self.max_sec = None
        self.hz: int = 50
        self.semantic_frame_rate: str = "25hz"
        self.segment_size: int = 20480
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048
        self.n_speakers: int = 300

    def save_configs(self, configs_path: str = None) -> None:
        path = configs_path or self.configs_path
        if path is None:
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "tts" not in data:
            data["tts"] = {}
        data["tts"]["custom"] = self.update_configs()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def update_configs(self):
        self.config = {
            "device": str(self.device),
            "is_half": self.is_half,
            "version": self.version,
            "t2s_weights_path": self.t2s_weights_path,
            "vits_weights_path": self.vits_weights_path,
            "hubert_base_path": self.hubert_base_path,
        }
        return self.config

    def update_version(self, version: str) -> None:
        self.version = version
        self.languages = self.v1_languages if self.version == "v1" else self.v2_languages

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, "-") + "\n"
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTS_Config) and self.configs_path == other.configs_path
