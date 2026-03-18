"""v1/v2 TorchScript export.

GPT-SoVITS v1/v2 모델을 TorchScript로 내보낸다.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torchaudio
from loguru import logger
from torch import IntTensor, LongTensor, Tensor, nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import setup_paths

setup_paths()
sys.path.insert(0, os.path.dirname(__file__))

from _t2s_torchscript import T2SModel, get_raw_t2s_model, spectrogram_torch
from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn
from process_ckpt import load_sovits_new
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from TTS_infer_pack.TTS_config import DictToAttrRecursive

from tools.utils.audio import load_audio

bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
hubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
cnhubert.hubert_base_path = hubert_base_path


@torch.jit.script
def build_phone_level_feature(res: Tensor, word2ph: IntTensor):
    phone_level_feature = []
    for i in range(word2ph.shape[0]):
        repeat_feature = res[i].repeat(word2ph[i].item(), 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature


class VitsModel(nn.Module):
    def __init__(self, vits_path, version=None, is_half=True, device="cpu"):
        super().__init__()
        dict_s2 = load_sovits_new(vits_path)
        self.hps = dict_s2["config"]

        if version is None:
            if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                self.hps["model"]["version"] = "v1"
            else:
                self.hps["model"]["version"] = "v2"
        else:
            if version in ["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"]:
                self.hps["model"]["version"] = version
            else:
                raise ValueError(f"Unsupported version: {version}")

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        self.vq_model.dec.remove_weight_norm()
        if is_half:
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(device)
        self.vq_model.eval()
        self.hann_window = torch.hann_window(
            self.hps.data.win_length, device=device, dtype=torch.float16 if is_half else torch.float32
        )

    def forward(self, text_seq, pred_semantic, ref_audio, speed=1.0, sv_emb=None):
        refer = spectrogram_torch(
            self.hann_window,
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return self.vq_model(pred_semantic, text_seq, refer, speed=speed, sv_emb=sv_emb)[0, 0]


class MyBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(MyBertModel, self).__init__()
        self.bert = bert_model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, word2ph: IntTensor
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        res = torch.cat(outputs[1][-3:-2], -1)[0][1:-1]
        return build_phone_level_feature(res, word2ph)


class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl = cnhubert.get_model().model

    def forward(self, ref_audio_16k) -> torch.Tensor:
        ssl_content = self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
        return ssl_content


class ExportSSLModel(torch.nn.Module):
    def __init__(self, ssl: SSLModel):
        super().__init__()
        self.ssl = ssl

    def forward(self, ref_audio: torch.Tensor):
        return self.ssl(ref_audio)

    @torch.jit.export
    def resample(self, ref_audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        audio = torchaudio.functional.resample(ref_audio, src_sr, dst_sr).float()
        return audio


class GPT_SoVITS(nn.Module):
    def __init__(self, t2s: T2SModel, vits: VitsModel):
        super().__init__()
        self.t2s = t2s
        self.vits = vits

    def forward(
        self,
        ssl_content: torch.Tensor,
        ref_audio_sr: torch.Tensor,
        ref_seq: Tensor,
        text_seq: Tensor,
        ref_bert: Tensor,
        text_bert: Tensor,
        top_k: LongTensor,
        speed=1.0,
    ):
        codes = self.vits.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompts = prompt_semantic.unsqueeze(0)

        pred_semantic = self.t2s(prompts, ref_seq, text_seq, ref_bert, text_bert, top_k)
        audio = self.vits(text_seq, pred_semantic, ref_audio_sr, speed)
        return audio


def _get_phones_and_bert(text: str, language: str, version: str):
    """텍스트에서 phone ID와 BERT 특성을 추출한다."""
    preprocessor = TextPreprocessor("cpu")
    return preprocessor.get_phones_and_bert(text, language, version)


def export(gpt_path, vits_path, ref_audio_path, ref_text, output_path, export_bert_and_ssl=False, device="cpu"):
    """v1/v2 GPT-SoVITS 모델을 TorchScript로 내보낸다."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info("디렉토리 생성: {}", output_path)
    else:
        logger.info("디렉토리 존재: {}", output_path)

    ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float()
    ssl = SSLModel()
    if export_bert_and_ssl:
        s = ExportSSLModel(torch.jit.trace(ssl, example_inputs=(ref_audio)))
        ssl_path = os.path.join(output_path, "ssl_model.pt")
        torch.jit.script(s).save(ssl_path)
        logger.info("ssl 모델 저장 완료")
    else:
        s = ExportSSLModel(ssl)

    logger.info("device: {}", device)

    ref_seq_id, ref_bert_T, ref_norm_text = _get_phones_and_bert(ref_text, "auto", "v2")
    ref_seq = torch.LongTensor([ref_seq_id]).to(device)
    ref_bert = ref_bert_T.T.to(ref_seq.device)
    text_seq_id, text_bert_T, norm_text = _get_phones_and_bert(
        "This is a simple example. The King and His Stories. Once there was a king.",
        "auto",
        "v2",
    )
    text_seq = torch.LongTensor([text_seq_id]).to(device)
    text_bert = text_bert_T.T.to(text_seq.device)

    ssl_content = ssl(ref_audio).to(device)

    vits = VitsModel(vits_path, device=device, is_half=False)
    vits.eval()

    dict_s1 = torch.load(gpt_path, weights_only=False)
    raw_t2s = get_raw_t2s_model(dict_s1).to(device)
    logger.info("get_raw_t2s_model 완료")
    logger.debug("t2s config: {}", raw_t2s.config)
    t2s_m = T2SModel(raw_t2s)
    t2s_m.eval()
    t2s = torch.jit.script(t2s_m).to(device)
    logger.info("t2s script 완료")

    logger.info("sampling_rate: {}", vits.hps.data.sampling_rate)
    gpt_sovits = GPT_SoVITS(t2s, vits).to(device)
    gpt_sovits.eval()

    ref_audio_sr = s.resample(ref_audio, 16000, 32000).to(device)

    torch._dynamo.mark_dynamic(ssl_content, 2)
    torch._dynamo.mark_dynamic(ref_audio_sr, 1)
    torch._dynamo.mark_dynamic(ref_seq, 1)
    torch._dynamo.mark_dynamic(text_seq, 1)
    torch._dynamo.mark_dynamic(ref_bert, 0)
    torch._dynamo.mark_dynamic(text_bert, 0)

    top_k = torch.LongTensor([5]).to(device)

    with torch.no_grad():
        gpt_sovits_export = torch.jit.trace(
            gpt_sovits, example_inputs=(ssl_content, ref_audio_sr, ref_seq, text_seq, ref_bert, text_bert, top_k)
        )

        gpt_sovits_path = os.path.join(output_path, "gpt_sovits_model.pt")
        gpt_sovits_export.save(gpt_sovits_path)
        logger.info("gpt_sovits 모델 저장 완료")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS v1/v2 TorchScript export")
    parser.add_argument("--gpt_model", required=True, help="GPT 모델 파일 경로")
    parser.add_argument("--sovits_model", required=True, help="SoVITS 모델 파일 경로")
    parser.add_argument("--ref_audio", required=True, help="참조 오디오 파일 경로")
    parser.add_argument("--ref_text", required=True, help="참조 텍스트")
    parser.add_argument("--output_path", required=True, help="출력 디렉토리")
    parser.add_argument("--export_common_model", action="store_true", help="Bert/SSL 모델도 export")
    parser.add_argument("--device", default="cpu", help="디바이스")

    args = parser.parse_args()
    export(
        gpt_path=args.gpt_model,
        vits_path=args.sovits_model,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output_path,
        device=args.device,
        export_bert_and_ssl=args.export_common_model,
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
