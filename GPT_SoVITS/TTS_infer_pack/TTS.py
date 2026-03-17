"""GPT-SoVITS TTS 추론 오케스트레이터.

v1/v2 합성은 직접 처리하고, v3/v4 vocoder 합성은 vocoder_v3v4 모듈에 위임한다.
"""
import gc
import math
import os
import random
import sys
import time
from copy import deepcopy
from typing import List, Tuple, Union

import ffmpeg
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import yaml
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor.cnhubert import CNHubert
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from sv import SV

from loguru import logger

try:
    from tools.audio_sr import AP_BWE
except ImportError:
    AP_BWE = None
from TTS_infer_pack.text_segmentation_method import splits
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from TTS_infer_pack.TTS_config import TTS_Config, DictToAttrRecursive

resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


class NO_PROMPT_ERROR(Exception):
    pass


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    raw_audio = input_audio.astype(np.int16).tobytes()
    input_stream = ffmpeg.input("pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1)
    output_stream = input_stream.filter("atempo", speed)
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )
    processed_audio = np.frombuffer(out, np.int16)
    return processed_audio


def set_seed(seed: int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    logger.debug("시드 설정: {}", seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed


class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config]):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs: TTS_Config = TTS_Config(configs)

        self.t2s_model: Text2SemanticLightningModule = None
        self.vits_model = None  # SynthesizerTrn 또는 SynthesizerTrnV3
        self.hubert_model: CNHubert = None
        self.vocoder = None  # v3/v4 전용
        self.sr_model = None  # AP_BWE (super resolution)
        self.sv_model = None  # SV (v2Pro 전용)
        self.sr_model_not_exist: bool = False
        self.is_v2pro: bool = False

        self.vocoder_configs: dict = {
            "sr": None,
            "T_ref": None,
            "T_chunk": None,
            "upsample_rate": None,
            "overlapped_len": None,
        }

        self._init_models()

        self.text_preprocessor: TextPreprocessor = TextPreprocessor(self.configs.device)

        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }

        self.stop_flag: bool = False
        self.precision: torch.dtype = torch.float16 if self.configs.is_half else torch.float32

    def _init_models(self):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_hubert_weights(self.configs.hubert_base_path)

    def init_hubert_weights(self, base_path: str):
        logger.info("CNHuBERT 가중치 로드: {}", base_path)
        self.hubert_model = CNHubert(base_path)
        self.hubert_model = self.hubert_model.eval()
        self.hubert_model = self.hubert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.hubert_model = self.hubert_model.half()

    def init_vits_weights(self, weights_path: str):
        self.configs.vits_weights_path = weights_path
        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(weights_path)
        if "Pro" in model_version:
            self.init_sv_model()

        dict_s2 = load_sovits_new(weights_path)
        hps = dict_s2["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps["model"]["version"] = "v2"
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps["model"]["version"] = "v1"
        else:
            hps["model"]["version"] = "v2"
        version = hps["model"]["version"]

        v3v4set = {"v3", "v4"}
        if model_version not in v3v4set:
            if "Pro" not in model_version:
                model_version = version
            else:
                hps["model"]["version"] = model_version
        else:
            hps["model"]["version"] = model_version

        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = hps["model"]["semantic_frame_rate"]
        kwargs = hps["model"]

        self.configs.update_version(model_version)

        if model_version not in v3v4set:
            # v1/v2/v2Pro — SynthesizerTrn 직접 로딩
            vits_model = SynthesizerTrn(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs,
            )
            self.configs.use_vocoder = False

            if if_lora_v3 == False:
                load_result = vits_model.load_state_dict(dict_s2["weight"], strict=False)
                logger.info("VITS 가중치 로드: {} ({})", weights_path, load_result)

            vits_model = vits_model.to(self.configs.device)
            vits_model = vits_model.eval()
            self.vits_model = vits_model
            if self.configs.is_half and str(self.configs.device) != "cpu":
                self.vits_model = self.vits_model.half()
        else:
            # v3/v4 — vocoder_v3v4 모듈에 위임
            from TTS_infer_pack import vocoder_v3v4

            vits_model = vocoder_v3v4.load_v3v4_vits(
                weights_path, dict_s2, hps, model_version, self.configs,
                self.configs.device, self.configs.is_half,
            )
            self.vits_model = vits_model
            self.configs.use_vocoder = True
            self.vocoder, self.vocoder_configs = vocoder_v3v4.init_vocoder(
                model_version, self.configs.device, self.configs.is_half, self.vocoder,
            )

        self.is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        self.configs.save_configs()

    def init_t2s_weights(self, weights_path: str):
        logger.info("Text2Semantic 가중치 로드: {}", weights_path)
        self.configs.t2s_weights_path = weights_path
        self.configs.save_configs()
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device, weights_only=False)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        self.t2s_model = t2s_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.t2s_model = self.t2s_model.half()

        codebook = t2s_model.model.ar_audio_embedding.weight.clone()
        mute_emb = codebook[self.configs.mute_tokens[self.configs.version]].unsqueeze(0)
        sim_matrix = F.cosine_similarity(mute_emb.float(), codebook.float(), dim=-1)
        self.configs.mute_emb_sim_matrix = sim_matrix

    def init_sr_model(self):
        if self.sr_model is not None:
            return
        try:
            self.sr_model = AP_BWE(self.configs.device, DictToAttrRecursive)
            self.sr_model_not_exist = False
        except (FileNotFoundError, TypeError):
            logger.warning("초해상도 모델 파라미터를 다운로드하지 않아 초해상도를 수행하지 않습니다")
            self.sr_model_not_exist = True

    def init_sv_model(self):
        if self.sv_model is not None:
            return
        self.sv_model = SV(self.configs.device, self.configs.is_half)

    def enable_half_precision(self, enable: bool = True, save: bool = True):
        if str(self.configs.device) == "cpu" and enable:
            logger.warning("CPU에서는 half precision을 지원하지 않습니다")
            return

        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        if save:
            self.configs.save_configs()
        if enable:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.half()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.half()
            if self.hubert_model is not None:
                self.hubert_model = self.hubert_model.half()
            if self.vocoder is not None:
                self.vocoder = self.vocoder.half()
        else:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.float()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.float()
            if self.hubert_model is not None:
                self.hubert_model = self.hubert_model.float()
            if self.vocoder is not None:
                self.vocoder = self.vocoder.float()

    def set_device(self, device: torch.device, save: bool = True):
        self.configs.device = device
        if save:
            self.configs.save_configs()
        if self.t2s_model is not None:
            self.t2s_model = self.t2s_model.to(device)
        if self.vits_model is not None:
            self.vits_model = self.vits_model.to(device)
        if self.hubert_model is not None:
            self.hubert_model = self.hubert_model.to(device)
        if self.vocoder is not None:
            self.vocoder = self.vocoder.to(device)
        if self.sr_model is not None:
            self.sr_model = self.sr_model.to(device)

    def set_ref_audio(self, ref_audio_path: str):
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def _set_ref_spec(self, ref_audio_path):
        spec_audio = self._get_ref_spec(ref_audio_path)
        if self.prompt_cache["refer_spec"] in [[], None]:
            self.prompt_cache["refer_spec"] = [spec_audio]
        else:
            self.prompt_cache["refer_spec"][0] = spec_audio

    def _get_ref_spec(self, ref_audio_path):
        data, raw_sr = sf.read(ref_audio_path, dtype="float32")
        raw_audio = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
        raw_audio = raw_audio.to(self.configs.device).float()
        self.prompt_cache["raw_audio"] = raw_audio
        self.prompt_cache["raw_sr"] = raw_sr

        if raw_sr != self.configs.sampling_rate:
            audio = raw_audio.to(self.configs.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)
            audio = resample(audio, raw_sr, self.configs.sampling_rate, self.configs.device)
        else:
            audio = raw_audio.to(self.configs.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)

        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        spec = spectrogram_torch(
            audio,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        if self.configs.is_half:
            spec = spec.half()
        if self.is_v2pro:
            audio = resample(audio, self.configs.sampling_rate, 16000, self.configs.device)
            if self.configs.is_half:
                audio = audio.half()
        else:
            audio = None
        return spec, audio

    def _set_prompt_semantic(self, ref_wav_path: str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("참조 오디오가 3~10초 범위 밖입니다, 교체해주세요")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.hubert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(
                1, 2
            )
            codes = self.vits_model.extract_latent(hubert_feature)
            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length: int = None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype: torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(
        self,
        data: list,
        prompt_data: dict = None,
        batch_size: int = 5,
        threshold: float = 0.75,
        split_bucket: bool = True,
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
    ):
        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                pos_end = min(pos + batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end - pos) // 2] / (batch.mean() + 1e-8)
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end - 1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1).to(
                        dtype=precision, device=device
                    )
                    all_phones = torch.LongTensor(prompt_data["phones"] + item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                else:
                    all_bert_features = item["bert_features"].to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list

            max_len = max(all_bert_max_len, all_phones_max_len)

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        length = len(sum(batch_index_list, []))
        _data = [None] * length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(self):
        self.stop_flag = True

    @torch.no_grad()
    def synthesize(self, inputs: dict):
        """Text to speech inference.

        Args:
            inputs: 합성 파라미터 딕셔너리.
        Yields:
            Tuple[int, np.ndarray]: (sampling_rate, audio_data)
        """
        self.stop_flag = False
        text: str = inputs.get("text", "")
        text_lang: str = inputs.get("text_lang", "")
        ref_audio_path: str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
        prompt_text: str = inputs.get("prompt_text", "")
        prompt_lang: str = inputs.get("prompt_lang", "")
        top_k: int = inputs.get("top_k", 15)
        top_p: float = inputs.get("top_p", 1)
        temperature: float = inputs.get("temperature", 1)
        text_split_method: str = inputs.get("text_split_method", "cut1")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        sample_steps = inputs.get("sample_steps", 32)
        super_sampling = inputs.get("super_sampling", False)
        streaming_mode = inputs.get("streaming_mode", False)
        overlap_length = inputs.get("overlap_length", 2)
        min_chunk_length = inputs.get("min_chunk_length", 16)
        fixed_length_chunk = inputs.get("fixed_length_chunk", False)
        chunk_split_thershold = 0.0

        if parallel_infer and not streaming_mode:
            logger.info("병렬 추론 모드 활성화됨")
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        elif not parallel_infer and streaming_mode and not self.configs.use_vocoder:
            logger.info("스트리밍 추론 모드 활성화됨")
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive
        elif streaming_mode and self.configs.use_vocoder:
            logger.warning("SoVITS v3/v4 모델은 스트리밍 추론을 지원하지 않아 분절 반환 모드로 전환됨")
            streaming_mode = False
            return_fragment = True
            if parallel_infer:
                self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
            else:
                self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched
        elif parallel_infer and streaming_mode:
            logger.warning("병렬 추론과 스트리밍 추론을 동시에 사용할 수 없어 병렬 추론 비활성화됨")
            parallel_infer = False
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive
        else:
            logger.info("기본 추론 모드 활성화됨")
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment and streaming_mode:
            logger.warning("스트리밍 추론은 분절 반환을 지원하지 않아 비활성화됨")
            return_fragment = False

        if (return_fragment or streaming_mode) and split_bucket:
            logger.warning("분절 반환/스트리밍 모드에서는 버킷 처리를 지원하지 않아 비활성화됨")
            split_bucket = False

        if split_bucket and speed_factor == 1.0 and not (self.configs.use_vocoder and parallel_infer):
            logger.info("버킷 처리 모드 활성화됨")
        elif speed_factor != 1.0:
            logger.warning("속도 조절 시 버킷 처리를 지원하지 않아 비활성화됨")
            split_bucket = False
        elif self.configs.use_vocoder and parallel_infer:
            logger.warning("병렬 추론 모드에서 SoVITS v3/v4는 버킷 처리를 지원하지 않아 비활성화됨")
            split_bucket = False
        else:
            logger.info("버킷 처리 모드 비활성화됨")

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if no_prompt_text and self.configs.use_vocoder:
            raise NO_PROMPT_ERROR("prompt_text cannot be empty when using SoVITS_V3/V4")

        if ref_audio_path in [None, ""] and (
            (self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError(
                "ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()"
            )

        ###### setting reference audio and prompt text preprocessing ########
        t0 = time.perf_counter()
        if (ref_audio_path is not None) and (
            ref_audio_path != self.prompt_cache["ref_audio_path"]
            or (self.is_v2pro and self.prompt_cache["refer_spec"][0][1] is None)
        ):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(self.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.prompt_cache["aux_ref_audio_paths"])):
            self.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    logger.warning("오디오 파일 없음, 건너뜀: {}", path)
                    continue
                self.prompt_cache["refer_spec"].append(self._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            logger.debug("실제 입력된 참고 텍스트: {}", prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text, prompt_lang, self.configs.version
                )
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = time.perf_counter()
        data: list = None
        if not (return_fragment or streaming_mode):
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return

            batch_index_list: list = None
            data, batch_index_list = self.to_batch(
                data,
                prompt_data=self.prompt_cache if not no_prompt_text else None,
                batch_size=batch_size,
                threshold=batch_threshold,
                split_bucket=split_bucket,
                device=self.configs.device,
                precision=self.precision,
            )
        else:
            logger.info("텍스트 분할")
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                logger.info("텍스트 BERT 특징 추출")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                        text, text_lang, self.configs.version
                    )
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(
                    batch_data,
                    prompt_data=self.prompt_cache if not no_prompt_text else None,
                    batch_size=batch_size,
                    threshold=batch_threshold,
                    split_bucket=False,
                    device=self.configs.device,
                    precision=self.precision,
                )
                return batch[0]

        t2 = time.perf_counter()
        try:
            logger.info("추론 시작")
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            is_first_package = True
            output_sr = self.configs.sampling_rate if not self.configs.use_vocoder else self.vocoder_configs["sr"]
            for item in data:
                t3 = time.perf_counter()
                if return_fragment or streaming_mode:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones: List[torch.LongTensor] = item["phones"]
                batch_phones_len: torch.LongTensor = item["phones_len"]
                all_phoneme_ids: torch.LongTensor = item["all_phones"]
                all_phoneme_lens: torch.LongTensor = item["all_phones_len"]
                all_bert_features: torch.LongTensor = item["all_bert_features"]
                norm_text: str = item["norm_text"]
                max_len = item["max_len"]

                logger.debug("프론트엔드 처리 후 텍스트(문장별): {}", norm_text)
                if no_prompt_text:
                    prompt = None
                else:
                    prompt = (
                        self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)
                    )

                refer_audio_spec = []

                sv_emb = [] if self.is_v2pro else None
                for spec, audio_tensor in self.prompt_cache["refer_spec"]:
                    spec = spec.to(dtype=self.precision, device=self.configs.device)
                    refer_audio_spec.append(spec)
                    if self.is_v2pro:
                        sv_emb.append(self.sv_model.compute_embedding3(audio_tensor))

                if not streaming_mode:
                    logger.info("의미 기반 토큰 예측")
                    pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_lens,
                        prompt,
                        all_bert_features,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=self.configs.hz * self.configs.max_sec,
                        max_len=max_len,
                        repetition_penalty=repetition_penalty,
                    )
                    t4 = time.perf_counter()
                    t_34 += t4 - t3

                    batch_audio_fragment = []

                    logger.info("오디오 생성")
                    if not self.configs.use_vocoder:
                        # v1/v2/v2Pro 합성
                        if speed_factor == 1.0:
                            logger.info("병렬 오디오 생성 중")
                            pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                            upsample_rate = math.prod(self.vits_model.upsample_rates)
                            audio_frag_idx = [
                                pred_semantic_list[i].shape[0] * 2 * upsample_rate
                                for i in range(0, len(pred_semantic_list))
                            ]
                            audio_frag_end_idx = [sum(audio_frag_idx[: i + 1]) for i in range(0, len(audio_frag_idx))]
                            all_pred_semantic = (
                                torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                            )
                            _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)

                            _batch_audio_fragment = self.vits_model.decode(
                                    all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                                ).detach()[0, 0, :]

                            audio_frag_end_idx.insert(0, 0)
                            batch_audio_fragment = [
                                _batch_audio_fragment[audio_frag_end_idx[i - 1] : audio_frag_end_idx[i]]
                                for i in range(1, len(audio_frag_end_idx))
                            ]
                        else:
                            for i, idx in enumerate(tqdm(idx_list)):
                                phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                                _pred_semantic = (
                                    pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                                )
                                audio_fragment = self.vits_model.decode(
                                        _pred_semantic, phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                                    ).detach()[0, 0, :]
                                batch_audio_fragment.append(audio_fragment)
                    else:
                        # v3/v4 vocoder 합성 — vocoder_v3v4 모듈에 위임
                        from TTS_infer_pack import vocoder_v3v4

                        if parallel_infer:
                            logger.info("병렬 오디오 생성 중")
                            audio_fragments = vocoder_v3v4.vocoder_synthesis_batched(
                                self.vits_model, self.vocoder, self.vocoder_configs,
                                self.prompt_cache, self.precision, self.configs,
                                idx_list, pred_semantic_list, batch_phones,
                                speed=speed_factor, sample_steps=sample_steps,
                                resample_fn=resample, sola_fn=self.sola_algorithm,
                            )
                            batch_audio_fragment.extend(audio_fragments)
                        else:
                            for i, idx in enumerate(tqdm(idx_list)):
                                phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                                _pred_semantic = (
                                    pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                                )
                                audio_fragment = vocoder_v3v4.vocoder_synthesis(
                                    self.vits_model, self.vocoder, self.vocoder_configs,
                                    self.prompt_cache, self.precision, self.configs,
                                    _pred_semantic, phones, refer_audio_spec,
                                    speed=speed_factor, sample_steps=sample_steps,
                                    resample_fn=resample,
                                )
                                batch_audio_fragment.append(audio_fragment)

                else:
                    # streaming mode (v1/v2 전용)
                    semantic_token_generator = self.t2s_model.model.infer_panel(
                        all_phoneme_ids[0].unsqueeze(0),
                        all_phoneme_lens,
                        prompt,
                        all_bert_features[0].unsqueeze(0),
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=self.configs.hz * self.configs.max_sec,
                        max_len=max_len,
                        repetition_penalty=repetition_penalty,
                        streaming_mode=True,
                        chunk_length=min_chunk_length,
                        mute_emb_sim_matrix=self.configs.mute_emb_sim_matrix if not fixed_length_chunk else None,
                        chunk_split_thershold=chunk_split_thershold,
                    )
                    t4 = time.perf_counter()
                    t_34 += t4 - t3
                    phones = batch_phones[0].unsqueeze(0).to(self.configs.device)
                    is_first_chunk = True

                    if not self.configs.use_vocoder:
                        upsample_rate = math.prod(self.vits_model.upsample_rates) * (
                            (2 if self.vits_model.semantic_frame_rate == "25hz" else 1) / speed_factor
                        )
                    else:
                        upsample_rate = self.vocoder_configs["upsample_rate"] * (
                            (3.875 if self.configs.version == "v3" else 4) / speed_factor
                        )

                    last_audio_chunk = None
                    last_latent = None
                    previous_tokens = []
                    overlap_len = overlap_length
                    overlap_size = math.ceil(overlap_length * upsample_rate)
                    for semantic_tokens, is_final in semantic_token_generator:

                        if semantic_tokens is None and last_audio_chunk is not None:
                            yield self.audio_postprocess(
                                    [[last_audio_chunk[-overlap_size:]]],
                                    output_sr,
                                    None,
                                    speed_factor,
                                    False,
                                    0.0,
                                    super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                                )
                            break

                        _semantic_tokens = semantic_tokens
                        logger.debug("semantic_tokens shape: {}", semantic_tokens.shape)

                        previous_tokens.append(semantic_tokens)
                        _semantic_tokens = torch.cat(previous_tokens, dim=-1)

                        if not is_first_chunk and semantic_tokens.shape[-1] < 10:
                            overlap_len = overlap_length + (10 - semantic_tokens.shape[-1])
                        else:
                            overlap_len = overlap_length

                        if not self.configs.use_vocoder:
                            token_padding_length = 0
                            audio_chunk, latent, latent_mask = self.vits_model.decode_streaming(
                                _semantic_tokens.unsqueeze(0),
                                phones, refer_audio_spec,
                                speed=speed_factor,
                                sv_emb=sv_emb,
                                result_length=semantic_tokens.shape[-1] + overlap_len if not is_first_chunk else None,
                                overlap_frames=last_latent[:, :, -overlap_len * (2 if self.vits_model.semantic_frame_rate == "25hz" else 1):]
                                if last_latent is not None else None,
                                padding_length=token_padding_length
                            )
                            audio_chunk = audio_chunk.detach()[0, 0, :]
                        else:
                            raise RuntimeError("SoVITS v3/v4 모델은 스트리밍 추론을 지원하지 않습니다")

                        if overlap_len > overlap_length:
                            audio_chunk = audio_chunk[-int((overlap_length + semantic_tokens.shape[-1]) * upsample_rate):]

                        audio_chunk_ = audio_chunk
                        if is_first_chunk and not is_final:
                            is_first_chunk = False
                            audio_chunk_ = audio_chunk_[:-overlap_size]
                        elif is_first_chunk and is_final:
                            is_first_chunk = False
                        elif not is_first_chunk and not is_final:
                            audio_chunk_ = self.sola_algorithm([last_audio_chunk, audio_chunk_], overlap_size)
                            audio_chunk_ = (
                                audio_chunk_[last_audio_chunk.shape[0] - overlap_size:-overlap_size] if not is_final
                                else audio_chunk_[last_audio_chunk.shape[0] - overlap_size:]
                            )

                        last_latent = latent
                        last_audio_chunk = audio_chunk
                        yield self.audio_postprocess(
                                [[audio_chunk_]],
                                output_sr,
                                None,
                                speed_factor,
                                False,
                                0.0,
                                super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                            )

                        if is_first_package:
                            logger.debug("first_package_delay: {:.3f}s", time.perf_counter() - t0)
                            is_first_package = False

                    yield output_sr, np.zeros(int(output_sr * fragment_interval), dtype=np.int16)

                t5 = time.perf_counter()
                t_45 += t5 - t4
                if return_fragment:
                    logger.debug("timing: preprocess={:.3f}s text={:.3f}s t2s={:.3f}s synth={:.3f}s", t1 - t0, t2 - t1, t4 - t3, t5 - t4)
                    yield self.audio_postprocess(
                        [batch_audio_fragment],
                        output_sr,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                        super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                    )
                elif streaming_mode:
                    ...
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield output_sr, np.zeros(int(output_sr), dtype=np.int16)
                    return

            if not (return_fragment or streaming_mode):
                logger.debug("timing: preprocess={:.3f}s text={:.3f}s t2s={:.3f}s synth={:.3f}s", t1 - t0, t2 - t1, t_34, t_45)
                if len(audio) == 0:
                    yield output_sr, np.zeros(int(output_sr), dtype=np.int16)
                    return
                yield self.audio_postprocess(
                    audio,
                    output_sr,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                    super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                )

        except Exception as e:
            logger.exception("TTS 추론 중 오류 발생")
            yield 16000, np.zeros(int(16000), dtype=np.int16)
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect()
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass

    def audio_postprocess(
        self,
        audio: List[torch.Tensor],
        sr: int,
        batch_index_list: list = None,
        speed_factor: float = 1.0,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
        super_sampling: bool = False,
    ) -> Tuple[int, np.ndarray]:
        if fragment_interval > 0:
            zero_wav = torch.zeros(
                int(self.configs.sampling_rate * fragment_interval), dtype=self.precision, device=self.configs.device
            )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio = torch.abs(audio_fragment).max()
                if max_audio > 1:
                    audio_fragment /= max_audio
                audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0) if fragment_interval > 0 else audio_fragment
                audio[i][j] = audio_fragment

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            audio = sum(audio, [])

        audio = torch.cat(audio, dim=0)

        if super_sampling:
            logger.info("오디오 초해상도")
            t1 = time.perf_counter()
            self.init_sr_model()
            if not self.sr_model_not_exist:
                audio, sr = self.sr_model(audio.unsqueeze(0), sr)
                max_audio = np.abs(audio).max()
                if max_audio > 1:
                    audio /= max_audio
            audio = (audio * 32768).astype(np.int16)
            t2 = time.perf_counter()
            logger.debug("초해상도 처리 소요 시간: {:.3f}s", t2 - t1)
        else:
            audio = audio.cpu().numpy()

        audio = (audio * 32768).astype(np.int16)

        return sr, audio

    def sola_algorithm(
        self,
        audio_fragments: List[torch.Tensor],
        overlap_len: int,
        search_len: int = 320,
    ):
        dtype = audio_fragments[0].dtype

        for i in range(len(audio_fragments) - 1):
            f1 = audio_fragments[i].float()
            f2 = audio_fragments[i + 1].float()
            w1 = f1[-overlap_len:]
            w2 = f2[:overlap_len + search_len]
            corr_norm = F.conv1d(w2.view(1, 1, -1), w1.view(1, 1, -1)).view(-1)

            corr_den = F.conv1d(w2.view(1, 1, -1) ** 2, torch.ones_like(w1).view(1, 1, -1)).view(-1) + 1e-8
            idx = (corr_norm / corr_den.sqrt()).argmax()

            logger.debug("seg_idx: {}", idx)

            f1_ = f1[:-overlap_len]
            audio_fragments[i] = f1_

            f2_ = f2[idx:]
            window = torch.hann_window((overlap_len) * 2, device=f1.device, dtype=f1.dtype)
            f2_[:overlap_len] = (
                window[:overlap_len] * f2_[:overlap_len]
                + window[overlap_len:] * f1[-overlap_len:]
            )

            audio_fragments[i + 1] = f2_

        return torch.cat(audio_fragments, 0).to(dtype)
