"""v3/v4 Vocoder 합성 모듈.

SynthesizerTrnV3 모델 로딩, BigVGAN/Generator vocoder 초기화,
그리고 vocoder 기반 오디오 합성 로직을 담당한다.
"""
from __future__ import annotations

import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from BigVGAN.bigvgan import BigVGAN
from module.mel_processing import mel_spectrogram_torch
from module.models import SynthesizerTrnV3, Generator
from loguru import logger
from peft import LoraConfig, get_peft_model
from process_ckpt import load_sovits_new

now_dir = os.getcwd()

# ── mel spectrogram 함수 ──

spec_min = -12
spec_max = 2


def normalize_mel(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denormalize_mel(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


# ── 모델 로딩 ──


def load_v3v4_vits(
    weights_path: str,
    dict_s2: dict,
    hps: dict,
    model_version: str,
    configs,
    device,
    is_half: bool,
) -> SynthesizerTrnV3:
    """SynthesizerTrnV3 모델을 로드한다. LoRA 가중치가 있으면 병합한다."""
    from process_ckpt import get_sovits_version_from_path_fast

    _, _, if_lora_v3 = get_sovits_version_from_path_fast(weights_path)

    path_sovits = configs.default_configs[model_version]["vits_weights_path"]

    if if_lora_v3 and not os.path.exists(path_sovits):
        raise FileExistsError(
            f"{path_sovits} SoVITS {model_version} 기본 모델 누락, LoRA 가중치를 로드할 수 없습니다"
        )

    kwargs = hps["model"]
    kwargs["version"] = model_version
    vits_model = SynthesizerTrnV3(
        configs.filter_length // 2 + 1,
        configs.segment_size // configs.hop_length,
        n_speakers=configs.n_speakers,
        **kwargs,
    )

    if "pretrained" not in weights_path and hasattr(vits_model, "enc_q"):
        del vits_model.enc_q

    if not if_lora_v3:
        load_result = vits_model.load_state_dict(dict_s2["weight"], strict=False)
        logger.info("VITS 가중치 로드: {} ({})", weights_path, load_result)
    else:
        pretrained_result = vits_model.load_state_dict(
            load_sovits_new(path_sovits)["weight"], strict=False
        )
        logger.info("VITS 기본 가중치 로드: {} ({})", path_sovits, pretrained_result)
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
        lora_result = vits_model.load_state_dict(dict_s2["weight"], strict=False)
        logger.info("LoRA 가중치 로드: {} ({})", weights_path, lora_result)
        vits_model.cfm = vits_model.cfm.merge_and_unload()

    vits_model = vits_model.to(device)
    vits_model = vits_model.eval()

    if is_half and str(device) != "cpu":
        vits_model = vits_model.half()

    return vits_model


def init_vocoder(
    version: str,
    device,
    is_half: bool,
    existing_vocoder=None,
) -> tuple:
    """v3/v4 vocoder를 초기화한다.

    Returns:
        (vocoder_model, vocoder_configs_dict)
    """
    if version == "v3":
        if existing_vocoder is not None and existing_vocoder.__class__.__name__ == "BigVGAN":
            return existing_vocoder, _vocoder_configs_v3()
        if existing_vocoder is not None:
            existing_vocoder.cpu()
            del existing_vocoder

        vocoder = BigVGAN.from_pretrained(
            "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
            use_cuda_kernel=False,
        )
        vocoder.remove_weight_norm()
        vocoder_configs = _vocoder_configs_v3()

    elif version == "v4":
        if existing_vocoder is not None and existing_vocoder.__class__.__name__ == "Generator":
            return existing_vocoder, _vocoder_configs_v4()
        if existing_vocoder is not None:
            existing_vocoder.cpu()
            del existing_vocoder

        vocoder = Generator(
            initial_channel=100,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4],
            gin_channels=0,
            is_bias=True,
        )
        vocoder.remove_weight_norm()
        state_dict_g = torch.load(
            "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
            map_location="cpu",
            weights_only=False,
        )
        load_result = vocoder.load_state_dict(state_dict_g)
        logger.info("vocoder 로드 완료: {}", load_result)
        vocoder_configs = _vocoder_configs_v4()
    else:
        raise ValueError(f"Unsupported vocoder version: {version}")

    vocoder = vocoder.eval()
    if is_half:
        vocoder = vocoder.half().to(device)
    else:
        vocoder = vocoder.to(device)

    return vocoder, vocoder_configs


def _vocoder_configs_v3() -> dict:
    return {
        "sr": 24000,
        "T_ref": 468,
        "T_chunk": 934,
        "upsample_rate": 256,
        "overlapped_len": 12,
    }


def _vocoder_configs_v4() -> dict:
    return {
        "sr": 48000,
        "T_ref": 500,
        "T_chunk": 1000,
        "upsample_rate": 480,
        "overlapped_len": 12,
    }


# ── 합성 ──


def vocoder_synthesis(
    vits_model,
    vocoder,
    vocoder_configs: dict,
    prompt_cache: dict,
    precision: torch.dtype,
    configs,
    semantic_tokens: torch.Tensor,
    phones: torch.Tensor,
    refer_audio_spec,
    speed: float = 1.0,
    sample_steps: int = 32,
    resample_fn=None,
) -> torch.Tensor:
    """v3/v4 vocoder 단일 합성."""
    prompt_semantic_tokens = prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(configs.device)
    prompt_phones = torch.LongTensor(prompt_cache["phones"]).unsqueeze(0).to(configs.device)
    raw_entry = prompt_cache["refer_spec"][0]
    if isinstance(raw_entry, tuple):
        raw_entry = raw_entry[0]
    refer_spec = raw_entry.to(dtype=precision, device=configs.device)

    fea_ref, ge = vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_spec)
    ref_audio: torch.Tensor = prompt_cache["raw_audio"]
    ref_sr = prompt_cache["raw_sr"]
    ref_audio = ref_audio.to(configs.device).float()
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)

    tgt_sr = 24000 if configs.version == "v3" else 32000
    if ref_sr != tgt_sr:
        if resample_fn is not None:
            ref_audio = resample_fn(ref_audio, ref_sr, tgt_sr, configs.device)
        else:
            import torchaudio

            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, tgt_sr)

    mel2 = mel_fn(ref_audio) if configs.version == "v3" else mel_fn_v4(ref_audio)
    mel2 = normalize_mel(mel2)
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :T_min]
    fea_ref = fea_ref[:, :, :T_min]
    T_ref = vocoder_configs["T_ref"]
    T_chunk = vocoder_configs["T_chunk"]
    if T_min > T_ref:
        mel2 = mel2[:, :, -T_ref:]
        fea_ref = fea_ref[:, :, -T_ref:]
        T_min = T_ref
    chunk_len = T_chunk - T_min

    mel2 = mel2.to(precision)
    fea_todo, ge = vits_model.decode_encp(semantic_tokens, phones, refer_spec, ge, speed)

    cfm_results = []
    idx = 0
    while True:
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break
        idx += chunk_len
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

        cfm_res = vits_model.cfm.inference(
            fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
        )
        cfm_res = cfm_res[:, :, mel2.shape[2] :]

        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]

        cfm_results.append(cfm_res)
    cfm_res = torch.cat(cfm_results, 2)
    cfm_res = denormalize_mel(cfm_res)

    with torch.inference_mode():
        wav_gen = vocoder(cfm_res)
        audio = wav_gen[0][0]

    return audio


def vocoder_synthesis_batched(
    vits_model,
    vocoder,
    vocoder_configs: dict,
    prompt_cache: dict,
    precision: torch.dtype,
    configs,
    idx_list: List[int],
    semantic_tokens_list: List[torch.Tensor],
    batch_phones: List[torch.Tensor],
    speed: float = 1.0,
    sample_steps: int = 32,
    resample_fn=None,
    sola_fn=None,
) -> List[torch.Tensor]:
    """v3/v4 vocoder 배치 병렬 합성."""
    prompt_semantic_tokens = prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(configs.device)
    prompt_phones = torch.LongTensor(prompt_cache["phones"]).unsqueeze(0).to(configs.device)
    raw_entry = prompt_cache["refer_spec"][0]
    if isinstance(raw_entry, tuple):
        raw_entry = raw_entry[0]
    refer_spec = raw_entry.to(dtype=precision, device=configs.device)

    fea_ref, ge = vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_spec)
    ref_audio: torch.Tensor = prompt_cache["raw_audio"]
    ref_sr = prompt_cache["raw_sr"]
    ref_audio = ref_audio.to(configs.device).float()
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)

    tgt_sr = 24000 if configs.version == "v3" else 32000
    if ref_sr != tgt_sr:
        if resample_fn is not None:
            ref_audio = resample_fn(ref_audio, ref_sr, tgt_sr, configs.device)
        else:
            import torchaudio

            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, tgt_sr)

    mel2 = mel_fn(ref_audio) if configs.version == "v3" else mel_fn_v4(ref_audio)
    mel2 = normalize_mel(mel2)
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :T_min]
    fea_ref = fea_ref[:, :, :T_min]
    T_ref = vocoder_configs["T_ref"]
    T_chunk = vocoder_configs["T_chunk"]
    if T_min > T_ref:
        mel2 = mel2[:, :, -T_ref:]
        fea_ref = fea_ref[:, :, -T_ref:]
        T_min = T_ref
    chunk_len = T_chunk - T_min

    mel2 = mel2.to(precision)

    # batched inference
    overlapped_len = vocoder_configs["overlapped_len"]
    feat_chunks = []
    feat_lens = []
    feat_list = []

    for i, idx in enumerate(idx_list):
        phones = batch_phones[i].unsqueeze(0).to(configs.device)
        semantic_tokens = semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0)
        feat, _ = vits_model.decode_encp(semantic_tokens, phones, refer_spec, ge, speed)
        feat_list.append(feat)
        feat_lens.append(feat.shape[2])

    feats = torch.cat(feat_list, 2)
    feats_padded = F.pad(feats, (overlapped_len, 0), "constant", 0)
    pos = 0
    padding_len = 0
    while True:
        if pos == 0:
            chunk = feats_padded[:, :, pos : pos + chunk_len]
        else:
            pos = pos - overlapped_len
            chunk = feats_padded[:, :, pos : pos + chunk_len]
        pos += chunk_len
        if chunk.shape[-1] == 0:
            break

        padding_len = chunk_len - chunk.shape[2]
        if padding_len != 0:
            chunk = F.pad(chunk, (0, padding_len), "constant", 0)
        feat_chunks.append(chunk)

    feat_chunks = torch.cat(feat_chunks, 0)
    bs = feat_chunks.shape[0]
    fea_ref = fea_ref.repeat(bs, 1, 1)
    fea = torch.cat([fea_ref, feat_chunks], 2).transpose(2, 1)
    pred_spec = vits_model.cfm.inference(
        fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
    )
    pred_spec = pred_spec[:, :, -chunk_len:]
    mel_channels = pred_spec.shape[1]
    pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(mel_channels, -1).unsqueeze(0)

    pred_spec = denormalize_mel(pred_spec)

    with torch.no_grad():
        wav_gen = vocoder(pred_spec)
        audio = wav_gen[0][0]

    audio_fragments = []
    upsample_rate = vocoder_configs["upsample_rate"]
    pos = 0

    while pos < audio.shape[-1]:
        audio_fragment = audio[pos : pos + chunk_len * upsample_rate]
        audio_fragments.append(audio_fragment)
        pos += chunk_len * upsample_rate

    if sola_fn is not None:
        audio = sola_fn(audio_fragments, overlapped_len * upsample_rate)
    else:
        audio = torch.cat(audio_fragments, 0)
    audio = audio[overlapped_len * upsample_rate : -padding_len * upsample_rate]

    audio_fragments = []
    for feat_len in feat_lens:
        audio_fragment = audio[: feat_len * upsample_rate]
        audio_fragments.append(audio_fragment)
        audio = audio[feat_len * upsample_rate :]

    return audio_fragments
