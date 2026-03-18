"""v3/v4 TorchScript export.

GPT-SoVITS v3/v4 모델을 TorchScript로 내보낸다.
- export_components: 개별 컴포넌트(vq_model, cfm, vocoder) TorchScript 내보내기
- export_end_to_end: 통합 엔드투엔드 모델 내보내기
"""
from __future__ import annotations

import argparse
import os
import sys

import librosa
import numpy as np
import soundfile
import torch
import torch._dynamo.config
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
# export 디렉토리 자체도 sys.path에 추가 (process_ckpt 등 import용)
sys.path.insert(0, os.path.dirname(__file__))

from _t2s_torchscript import T2SModel, get_raw_t2s_model, spectrogram_torch
from f5_tts.model.backbones.dit import DiT
from feature_extractor import cnhubert
from module import commons
from module.mel_processing import mel_spectrogram_torch
from module.models_onnx import CFM, Generator, SynthesizerTrnV3
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from TTS_infer_pack.TTS_config import DictToAttrRecursive

from tools.utils.audio import load_audio

is_half = True
device = "cuda" if torch.cuda.is_available() else "cpu"
now_dir = os.getcwd()

# ── Mel spectrogram 유틸 ──

def mel_fn(x):
    return mel_spectrogram_torch(
        x, n_fft=1024, win_size=1024, hop_size=256,
        num_mels=100, sampling_rate=24000, fmin=0, fmax=None, center=False,
    )


def mel_fn_v4(x):
    return mel_spectrogram_torch(
        x, n_fft=1280, win_size=1280, hop_size=320,
        num_mels=100, sampling_rate=32000, fmin=0, fmax=None, center=False,
    )


@torch.jit.script
def normalize_mel(x):
    spec_min = -12
    spec_max = 2
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denormalize_mel(x):
    spec_min = -12
    spec_max = 2
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


# ── Export 래퍼 모듈 ──


class MelSpectrgram(torch.nn.Module):
    def __init__(
        self,
        dtype,
        device,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super().__init__()
        self.hann_window = torch.hann_window(win_size).to(device=device, dtype=dtype)
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = torch.from_numpy(mel).to(dtype=dtype, device=device)
        self.n_fft: int = n_fft
        self.hop_size: int = hop_size
        self.win_size: int = win_size
        self.center: bool = center

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec


class ExportDitBlocks(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        self.transformer_blocks = dit.transformer_blocks
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out
        self.depth = dit.depth

    def forward(self, x, t, mask, rope):
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=(rope, 1.0))
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output


class ExportDitEmbed(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        self.time_embed = dit.time_embed
        self.d_embed = dit.d_embed
        self.text_embed = dit.text_embed
        self.input_embed = dit.input_embed
        self.rotary_embed = dit.rotary_embed
        self.rotary_embed.inv_freq.to(device)

    def forward(
        self,
        x0: torch.Tensor,
        cond0: torch.Tensor,
        x_lens: torch.Tensor,
        time: torch.Tensor,
        dt_base_bootstrap: torch.Tensor,
        text0: torch.Tensor,
    ):
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = commons.sequence_mask(x_lens, max_length=x.size(1)).to(x.device)

        t = self.time_embed(time) + self.d_embed(dt_base_bootstrap)
        text_embed = self.text_embed(text, x.shape[1])
        rope_t = torch.arange(x.shape[1], device=device)
        rope, _ = self.rotary_embed(rope_t)
        x = self.input_embed(x, cond, text_embed)
        return x, t, mask, rope


class ExportDiT(torch.nn.Module):
    def __init__(self, dit: DiT):
        super().__init__()
        if dit is not None:
            self.embed = ExportDitEmbed(dit)
            self.blocks = ExportDitBlocks(dit)
        else:
            self.embed = None
            self.blocks = None

    def forward(
        self,
        x0: torch.Tensor,
        cond0: torch.Tensor,
        x_lens: torch.Tensor,
        time: torch.Tensor,
        dt_base_bootstrap: torch.Tensor,
        text0: torch.Tensor,
    ):
        x, t, mask, rope = self.embed(x0, cond0, x_lens, time, dt_base_bootstrap, text0)
        output = self.blocks(x, t, mask, rope)
        return output


class ExportCFM(torch.nn.Module):
    def __init__(self, cfm: CFM):
        super().__init__()
        self.cfm = cfm

    def forward(
        self,
        fea_ref: torch.Tensor,
        fea_todo_chunk: torch.Tensor,
        mel2: torch.Tensor,
        sample_steps: torch.LongTensor,
    ):
        T_min = fea_ref.size(2)
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        cfm_res = self.cfm(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps)
        cfm_res = cfm_res[:, :, mel2.shape[2] :]
        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]
        return cfm_res, fea_ref, mel2


class ExportGPTSovitsHalf(torch.nn.Module):
    """v3 export 모듈 (24kHz, BigVGAN)."""

    def __init__(self, hps, t2s_m: T2SModel, vq_model: SynthesizerTrnV3):
        super().__init__()
        self.hps = hps
        self.t2s_m = t2s_m
        self.vq_model = vq_model
        self.mel2 = MelSpectrgram(
            dtype=torch.float32,
            device=device,
            n_fft=1024,
            num_mels=100,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=None,
            center=False,
        )
        self.filter_length: int = hps.data.filter_length
        self.sampling_rate: int = hps.data.sampling_rate
        self.hop_length: int = hps.data.hop_length
        self.win_length: int = hps.data.win_length
        self.hann_window = torch.hann_window(self.win_length, device=device, dtype=torch.float32)

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0,
        phoneme_ids1,
        bert1,
        bert2,
        top_k,
    ):
        refer = spectrogram_torch(
            self.hann_window,
            ref_audio_32k,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).to(ssl_content.dtype)

        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0)

        pred_semantic = self.t2s_m(prompt, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k)

        ge = self.vq_model.create_ge(refer)

        prompt_ = prompt.unsqueeze(0)
        fea_ref = self.vq_model(prompt_, phoneme_ids0, ge)

        ref_24k = torchaudio.functional.resample(ref_audio_32k, 32000, 24000).float()
        mel2 = normalize_mel(self.mel2(ref_24k)).to(ssl_content.dtype)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468

        fea_todo = self.vq_model(pred_semantic, phoneme_ids1, ge)

        return fea_ref, fea_todo, mel2


class ExportGPTSovitsV4Half(torch.nn.Module):
    """v4 export 모듈 (32kHz, HiFi-GAN)."""

    def __init__(self, hps, t2s_m: T2SModel, vq_model: SynthesizerTrnV3):
        super().__init__()
        self.hps = hps
        self.t2s_m = t2s_m
        self.vq_model = vq_model
        self.mel2 = MelSpectrgram(
            dtype=torch.float32,
            device=device,
            n_fft=1280,
            num_mels=100,
            sampling_rate=32000,
            hop_size=320,
            win_size=1280,
            fmin=0,
            fmax=None,
            center=False,
        )
        self.filter_length: int = hps.data.filter_length
        self.sampling_rate: int = hps.data.sampling_rate
        self.hop_length: int = hps.data.hop_length
        self.win_length: int = hps.data.win_length
        self.hann_window = torch.hann_window(self.win_length, device=device, dtype=torch.float32)

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0,
        phoneme_ids1,
        bert1,
        bert2,
        top_k,
    ):
        refer = spectrogram_torch(
            self.hann_window,
            ref_audio_32k,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).to(ssl_content.dtype)

        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0)

        pred_semantic = self.t2s_m(prompt, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k)

        ge = self.vq_model.create_ge(refer)

        prompt_ = prompt.unsqueeze(0)
        fea_ref = self.vq_model(prompt_, phoneme_ids0, ge)

        mel2 = normalize_mel(self.mel2(ref_audio_32k)).to(ssl_content.dtype)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        if T_min > 500:
            mel2 = mel2[:, :, -500:]
            fea_ref = fea_ref[:, :, -500:]
            T_min = 500

        fea_todo = self.vq_model(pred_semantic, phoneme_ids1, ge)

        return fea_ref, fea_todo, mel2


class GPTSoVITSV3(torch.nn.Module):
    """v3 엔드투엔드 모델 (BigVGAN vocoder, 24kHz, chunk=934, upsample=256)."""

    def __init__(self, gpt_sovits_half, cfm, bigvgan):
        super().__init__()
        self.gpt_sovits_half = gpt_sovits_half
        self.cfm = cfm
        self.bigvgan = bigvgan

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0: torch.LongTensor,
        phoneme_ids1: torch.LongTensor,
        bert1,
        bert2,
        top_k: torch.LongTensor,
        sample_steps: torch.LongTensor,
    ):
        fea_ref, fea_todo, mel2 = self.gpt_sovits_half(
            ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k
        )
        chunk_len = 934 - fea_ref.shape[2]
        wav_gen_list = []
        idx = 0
        fea_todo = fea_todo[:, :, :-5]
        wav_gen_length = fea_todo.shape[2] * 256
        while 1:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break

            # 내보낸 모델은 shape가 달라지면 재컴파일되어 ~10초 지연이 발생하므로,
            # 0으로 패딩하여 shape를 일정하게 유지한다.
            complete_len = chunk_len - fea_todo_chunk.shape[-1]
            if complete_len != 0:
                fea_todo_chunk = torch.cat(
                    [
                        fea_todo_chunk,
                        torch.zeros(1, 512, complete_len).to(fea_todo_chunk.device).to(fea_todo_chunk.dtype),
                    ],
                    2,
                )

            cfm_res, fea_ref, mel2 = self.cfm(fea_ref, fea_todo_chunk, mel2, sample_steps)
            idx += chunk_len

            cfm_res = denormalize_mel(cfm_res)
            bigvgan_res = self.bigvgan(cfm_res)
            wav_gen_list.append(bigvgan_res)

        wav_gen = torch.cat(wav_gen_list, 2)
        return wav_gen[0][0][:wav_gen_length]


class GPTSoVITSV4(torch.nn.Module):
    """v4 엔드투엔드 모델 (HiFi-GAN vocoder, 48kHz, chunk=1000, upsample=480)."""

    def __init__(self, gpt_sovits_half, cfm, hifigan):
        super().__init__()
        self.gpt_sovits_half = gpt_sovits_half
        self.cfm = cfm
        self.hifigan = hifigan

    def forward(
        self,
        ssl_content,
        ref_audio_32k: torch.FloatTensor,
        phoneme_ids0: torch.LongTensor,
        phoneme_ids1: torch.LongTensor,
        bert1,
        bert2,
        top_k: torch.LongTensor,
        sample_steps: torch.LongTensor,
    ):
        fea_ref, fea_todo, mel2 = self.gpt_sovits_half(
            ssl_content, ref_audio_32k, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k
        )
        chunk_len = 1000 - fea_ref.shape[2]
        wav_gen_list = []
        idx = 0
        fea_todo = fea_todo[:, :, :-10]
        wav_gen_length = fea_todo.shape[2] * 480
        while 1:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break

            # 내보낸 모델은 shape가 달라지면 재컴파일되어 ~10초 지연이 발생하므로,
            # 0으로 패딩하여 shape를 일정하게 유지한다.
            complete_len = chunk_len - fea_todo_chunk.shape[-1]
            if complete_len != 0:
                fea_todo_chunk = torch.cat(
                    [
                        fea_todo_chunk,
                        torch.zeros(1, 512, complete_len).to(fea_todo_chunk.device).to(fea_todo_chunk.dtype),
                    ],
                    2,
                )

            cfm_res, fea_ref, mel2 = self.cfm(fea_ref, fea_todo_chunk, mel2, sample_steps)
            idx += chunk_len

            cfm_res = denormalize_mel(cfm_res)
            hifigan_res = self.hifigan(cfm_res)
            wav_gen_list.append(hifigan_res)

        wav_gen = torch.cat(wav_gen_list, 2)
        return wav_gen[0][0][:wav_gen_length]


class Sovits:
    """SoVITS 모델 컨테이너 (export용)."""

    def __init__(self, vq_model: SynthesizerTrnV3, cfm: CFM, hps):
        self.vq_model = vq_model
        self.hps = hps
        cfm.estimator = ExportDiT(cfm.estimator)
        self.cfm = cfm


# ── Internal helpers ──

v3v4set = {"v3", "v4"}


def _compute_spectrogram(hps, audio_path: str) -> torch.Tensor:
    """참조 오디오에서 스펙트로그램을 계산한다."""
    audio = load_audio(audio_path, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    peak = audio.abs().max()
    if peak > 1:
        audio /= min(2, peak)
    audio_norm = audio.unsqueeze(0)
    hann_window = torch.hann_window(hps.data.win_length)
    spec = spectrogram_torch(
        hann_window,
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def _get_phones_and_bert(text: str, language: str, version: str):
    """텍스트에서 phone ID와 BERT 특성을 추출한다."""
    preprocessor = TextPreprocessor(device)
    return preprocessor.get_phones_and_bert(text, language, version)


def _load_sovits_for_export(sovits_path: str) -> Sovits:
    """SoVITS 모델을 export용으로 로드한다."""
    path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    if if_lora_v3 and not is_exist_s2gv3:
        logger.info("SoVITS V3 기본 모델이 없어 해당 LoRA 가중치를 로드할 수 없습니다")

    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    if model_version in v3v4set:
        hps.model.version = model_version

    logger.info("hps: {}", hps)

    vq_model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    model_version = hps.model.version
    logger.info("모델 버전: {}", model_version)

    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    vq_model.eval()

    cfm = vq_model.cfm
    del vq_model.cfm

    return Sovits(vq_model, cfm, hps)


def _init_bigvgan():
    """BigVGAN vocoder (v3용)를 초기화한다."""
    global bigvgan_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if is_half:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def _init_hifigan():
    """HiFi-GAN vocoder (v4용)를 초기화한다."""
    global hifigan_model
    hifigan_model = Generator(
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
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,), map_location="cpu"
    )
    logger.info("loading vocoder: {}", hifigan_model.load_state_dict(state_dict_g))
    if is_half:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


def _trace_cfm(
    e_cfm: ExportCFM,
    mu: torch.Tensor,
    x_lens: torch.LongTensor,
    prompt: torch.Tensor,
    n_timesteps: torch.IntTensor,
    output_dir: str,
    temperature=1.0,
):
    """CFM estimator를 trace하고 저장한다."""
    cfm = e_cfm.cfm

    B, T = mu.size(0), mu.size(1)
    x = torch.randn([B, cfm.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
    logger.debug("x: {} {}", x.shape, x.dtype)
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x, dtype=mu.dtype)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0.0
    mu = mu.transpose(2, 1)

    ntimestep = int(n_timesteps)

    t = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    d = torch.tensor(1.0 / ntimestep, dtype=x.dtype, device=x.device)

    t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
    d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d

    logger.debug(
        "cfm input shapes: {} {} {} {} {} {}",
        x.shape, prompt_x.shape, x_lens.shape, t_tensor.shape, d_tensor.shape, mu.shape,
    )
    logger.debug(
        "cfm input dtypes: {} {} {} {} {} {}",
        x.dtype, prompt_x.dtype, x_lens.dtype, t_tensor.dtype, d_tensor.dtype, mu.dtype,
    )

    estimator: ExportDiT = torch.jit.trace(
        cfm.estimator,
        optimize=True,
        example_inputs=(x, prompt_x, x_lens, t_tensor, d_tensor, mu),
    )
    estimator.save(os.path.join(output_dir, "estimator.pt"))
    logger.info("estimator 저장 완료")

    cfm.estimator = estimator
    export_cfm = torch.jit.script(e_cfm)
    export_cfm.save(os.path.join(output_dir, "cfm.pt"))
    return export_cfm


# ── Export 함수 ──


def export_components(
    ref_wav_path: str,
    ref_wav_text: str,
    sovits_path: str,
    gpt_path: str,
    version: str = "v3",
    output_dir: str = "onnx/ad",
):
    """개별 컴포넌트(vq_model, cfm, vocoder)를 TorchScript로 내보낸다.

    Phase 1: vq_model, cfm, bigvgan/hifigan 각각을 trace/script하여 저장한다.
    export_end_to_end()에서 이 결과를 조립한다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if version == "v3":
        sovits = _load_sovits_for_export(sovits_path)
        _init_bigvgan()
    else:
        sovits = _load_sovits_for_export(sovits_path)
        _init_hifigan()

    dict_s1 = torch.load(gpt_path)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    logger.info("get_raw_t2s_model 완료")
    logger.debug("t2s config: {}", raw_t2s.config)

    if is_half:
        raw_t2s = raw_t2s.half().to(device)

    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    script_t2s = torch.jit.script(t2s_m).to(device)

    hps = sovits.hps
    dtype = torch.float16 if is_half else torch.float32
    refer = _compute_spectrogram(hps, ref_wav_path).to(device).to(dtype)

    # SSL content 추출
    cnhubert.hubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    ssl_model = cnhubert.get_model()
    if is_half:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)

        if is_half:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = sovits.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    phones1, bert1, norm_text1 = _get_phones_and_bert(ref_wav_text, "auto", "v3")
    phones2, bert2, norm_text2 = _get_phones_and_bert(
        "This is a simple example. The King and His Stories. Once there was a king.",
        "auto",
        "v3",
    )
    phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
    phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)

    top_k = torch.LongTensor([15]).to(device)

    bert1 = bert1.T.to(device)
    bert2 = bert2.T.to(device)
    logger.debug(
        "dtypes: {} {} {} {} {} {}",
        prompt.dtype, phoneme_ids0.dtype, phoneme_ids1.dtype, bert1.dtype, bert2.dtype, top_k.dtype,
    )

    pred_semantic = t2s_m(prompt, phoneme_ids0, phoneme_ids1, bert1, bert2, top_k)

    ge = sovits.vq_model.create_ge(refer)
    prompt_ = prompt.unsqueeze(0)

    torch._dynamo.mark_dynamic(prompt_, 2)
    torch._dynamo.mark_dynamic(phoneme_ids0, 1)

    fea_ref = sovits.vq_model(prompt_, phoneme_ids0, ge)

    inputs = {
        "forward": (prompt_, phoneme_ids0, ge),
        "extract_latent": ssl_content,
        "create_ge": refer,
    }

    trace_vq_model = torch.jit.trace_module(sovits.vq_model, inputs, optimize=True)
    trace_vq_model.save(os.path.join(output_dir, "vq_model.pt"))
    logger.info("vq_model 저장 완료")

    vq_model = trace_vq_model

    if version == "v3":
        gpt_sovits_half = ExportGPTSovitsHalf(sovits.hps, script_t2s, trace_vq_model)
        torch.jit.script(gpt_sovits_half).save(os.path.join(output_dir, "gpt_sovits_v3_half.pt"))
    else:
        gpt_sovits_half = ExportGPTSovitsV4Half(sovits.hps, script_t2s, trace_vq_model)
        torch.jit.script(gpt_sovits_half).save(os.path.join(output_dir, "gpt_sovits_v4_half.pt"))
    logger.info("gpt_sovits_half 저장 완료")

    ref_audio, sr = torchaudio.load(ref_wav_path)
    ref_audio = ref_audio.to(device).float()
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)
    tgt_sr = 24000 if version == "v3" else 32000
    if sr != tgt_sr:
        ref_audio = torchaudio.functional.resample(ref_audio, sr, tgt_sr).float()
    mel2 = mel_fn(ref_audio) if version == "v3" else mel_fn_v4(ref_audio)
    mel2 = normalize_mel(mel2)
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    fea_ref = fea_ref[:, :, :T_min]
    logger.debug("fea_ref: {} T_min={}", fea_ref.shape, T_min)
    Tref = 468 if version == "v3" else 500
    Tchunk = 934 if version == "v3" else 1000
    if T_min > Tref:
        mel2 = mel2[:, :, -Tref:]
        fea_ref = fea_ref[:, :, -Tref:]
        T_min = Tref
    chunk_len = Tchunk - T_min
    mel2 = mel2.to(dtype)

    fea_todo = vq_model(pred_semantic, phoneme_ids1, ge)

    cfm_results = []
    idx = 0
    sample_steps = torch.LongTensor([8]).to(device)
    export_cfm_ = ExportCFM(sovits.cfm)
    while 1:
        logger.debug("idx: {}", idx)
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break

        if idx == 0:
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            export_cfm_ = _trace_cfm(
                export_cfm_,
                fea,
                torch.LongTensor([fea.size(1)]).to(fea.device),
                mel2,
                sample_steps,
                output_dir,
            )

        idx += chunk_len

        cfm_res, fea_ref, mel2 = export_cfm_(fea_ref, fea_todo_chunk, mel2, sample_steps)
        cfm_results.append(cfm_res)

    cmf_res = torch.cat(cfm_results, 2)
    cmf_res = denormalize_mel(cmf_res).to(device)
    logger.debug("cmf_res: {} {}", cmf_res.shape, cmf_res.dtype)

    with torch.inference_mode():
        cmf_res_rand = torch.randn(1, 100, 934).to(device).to(dtype)
        torch._dynamo.mark_dynamic(cmf_res_rand, 2)
        if version == "v3":
            bigvgan_model_ = torch.jit.trace(bigvgan_model, optimize=True, example_inputs=(cmf_res_rand,))
            bigvgan_model_.save(os.path.join(output_dir, "bigvgan_model.pt"))
            wav_gen = bigvgan_model(cmf_res)
        else:
            hifigan_model_ = torch.jit.trace(hifigan_model, optimize=True, example_inputs=(cmf_res_rand,))
            hifigan_model_.save(os.path.join(output_dir, "hifigan_model.pt"))
            wav_gen = hifigan_model(cmf_res)

        logger.debug("wav_gen: {} {}", wav_gen.shape, wav_gen.dtype)
        audio = wav_gen[0][0].cpu().detach().numpy()

    sr = 24000 if version == "v3" else 48000
    out_path = os.path.join(output_dir, "out.export.wav")
    soundfile.write(out_path, (audio * 32768).astype(np.int16), sr)
    logger.info("export_components 완료: {}", output_dir)


def export_end_to_end(
    sovits_path: str,
    gpt_path: str,
    version: str = "v3",
    output_dir: str = "onnx/ad",
):
    """사전 export된 컴포넌트를 조립하여 엔드투엔드 모델을 만든다.

    export_components()를 먼저 실행하여 개별 컴포넌트를 생성해야 한다.
    """
    if version == "v3":
        sovits = _load_sovits_for_export(sovits_path)
    else:
        sovits = _load_sovits_for_export(sovits_path)

    sovits.cfm = None

    cfm = torch.jit.load(os.path.join(output_dir, "cfm.pt"), map_location=device)
    cfm = cfm.half().to(device)
    cfm.eval()
    logger.info("cfm 로드 완료")

    dict_s1 = torch.load(gpt_path)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    logger.info("get_raw_t2s_model 완료")
    if is_half:
        raw_t2s = raw_t2s.half().to(device)
    t2s_m = T2SModel(raw_t2s).half().to(device)
    t2s_m.eval()
    t2s_m = torch.jit.script(t2s_m).to(device)
    t2s_m.eval()
    logger.info("t2s_m 로드 완료")

    vq_model: torch.jit.ScriptModule = torch.jit.load(
        os.path.join(output_dir, "vq_model.pt"), map_location=device
    )
    vq_model.eval()
    logger.info("vq_model 로드 완료")

    if version == "v3":
        gpt_sovits_v3_half = ExportGPTSovitsHalf(sovits.hps, t2s_m, vq_model)
        logger.info("gpt_sovits_v3_half 조립 완료")

        bigvgan_model = torch.jit.load(os.path.join(output_dir, "bigvgan_model.pt"))
        bigvgan_model = bigvgan_model.half().cuda()
        bigvgan_model.eval()
        logger.info("bigvgan 로드 완료")

        gpt_sovits_v3 = GPTSoVITSV3(gpt_sovits_v3_half, cfm, bigvgan_model)
        gpt_sovits_v3 = torch.jit.script(gpt_sovits_v3)
        gpt_sovits_v3.save(os.path.join(output_dir, "gpt_sovits_v3.pt"))
        gpt_sovits_v3 = gpt_sovits_v3.half().to(device)
        gpt_sovits_v3.eval()
        logger.info("gpt_sovits_v3 저장 완료")
    else:
        gpt_sovits_v4_half = ExportGPTSovitsV4Half(sovits.hps, t2s_m, vq_model)
        logger.info("gpt_sovits_v4_half 조립 완료")

        hifigan_model = torch.jit.load(os.path.join(output_dir, "hifigan_model.pt"))
        hifigan_model = hifigan_model.half().cuda()
        hifigan_model.eval()
        logger.info("hifigan 로드 완료")

        gpt_sovits_v4 = GPTSoVITSV4(gpt_sovits_v4_half, cfm, hifigan_model)
        gpt_sovits_v4 = torch.jit.script(gpt_sovits_v4)
        gpt_sovits_v4.save(os.path.join(output_dir, "gpt_sovits_v4.pt"))
        logger.info("gpt_sovits_v4 저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS v3/v4 TorchScript export")
    parser.add_argument("--version", choices=["v3", "v4"], required=True)
    parser.add_argument("--sovits-weights", required=True, help="SoVITS 모델 가중치 경로")
    parser.add_argument("--gpt-weights", required=True, help="GPT 모델 가중치 경로")
    parser.add_argument("--ref-audio", required=True, help="참조 오디오 경로")
    parser.add_argument("--ref-text", required=True, help="참조 텍스트")
    parser.add_argument("--output-dir", default="onnx/ad", help="출력 디렉토리")
    parser.add_argument("--phase", choices=["components", "e2e", "all"], default="all", help="실행 단계")

    args = parser.parse_args()

    with torch.no_grad():
        if args.phase in ("components", "all"):
            export_components(
                ref_wav_path=args.ref_audio,
                ref_wav_text=args.ref_text,
                sovits_path=args.sovits_weights,
                gpt_path=args.gpt_weights,
                version=args.version,
                output_dir=args.output_dir,
            )
        if args.phase in ("e2e", "all"):
            export_end_to_end(
                sovits_path=args.sovits_weights,
                gpt_path=args.gpt_weights,
                version=args.version,
                output_dir=args.output_dir,
            )
