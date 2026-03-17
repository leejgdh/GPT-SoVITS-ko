"""T2S TorchScript 래퍼.

Text-to-Semantic 모델의 TorchScript 호환 클래스와 유틸리티 함수들.
v1/v2 및 v3/v4 export 스크립트에서 공유한다.
"""
from __future__ import annotations

from typing import Optional

import torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F


def get_raw_t2s_model(dict_s1) -> Text2SemanticLightningModule:
    config = dict_s1["config"]
    config["model"]["dropout"] = float(config["model"]["dropout"])
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = t2s_model.eval()
    return t2s_model


@torch.jit.script
def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.jit.script
def multinomial_sample_one_no_sync(probs_sort):
    """CUDA 동기화 없이 multinomial 샘플링을 수행한다."""
    q = torch.empty_like(probs_sort).exponential_(1.0)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


@torch.jit.script
def sample(
    logits,
    previous_tokens,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.35,
):
    probs = logits_to_probs(
        logits=logits,
        previous_tokens=previous_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.jit.script
def spectrogram_torch(
    hann_window: Tensor, y: Tensor, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, center: bool = False
):
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


@torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1: float,
        norm_w2,
        norm_b2,
        norm_eps2: float,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        q, k, v = F.linear(self.to_mask(x, padding_mask), self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache

    def decode_next_token(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: list[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(self, x: torch.Tensor, attn_mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        k_cache: list[torch.Tensor] = []
        v_cache: list[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(self, x: torch.Tensor, k_cache: list[torch.Tensor], v_cache: list[torch.Tensor]):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i])
        return x, k_cache, v_cache


class T2SModel(nn.Module):
    def __init__(self, raw_t2s: Text2SemanticLightningModule):
        super(T2SModel, self).__init__()
        self.model_dim = raw_t2s.model.model_dim
        self.embedding_dim = raw_t2s.model.embedding_dim
        self.num_head = raw_t2s.model.num_head
        self.num_layers = raw_t2s.model.num_layers
        self.vocab_size = raw_t2s.model.vocab_size
        self.phoneme_vocab_size = raw_t2s.model.phoneme_vocab_size
        self.EOS: int = int(raw_t2s.model.EOS)
        self.norm_first = raw_t2s.model.norm_first
        assert self.EOS == self.vocab_size - 1
        self.hz = 50

        self.bert_proj = raw_t2s.model.bert_proj
        self.ar_text_embedding = raw_t2s.model.ar_text_embedding
        self.ar_text_position = raw_t2s.model.ar_text_position
        self.ar_audio_embedding = raw_t2s.model.ar_audio_embedding
        self.ar_audio_position = raw_t2s.model.ar_audio_position

        blocks = []
        h = raw_t2s.model.h

        for i in range(self.num_layers):
            layer = h.layers[i]
            t2smlp = T2SMLP(layer.linear1.weight, layer.linear1.bias, layer.linear2.weight, layer.linear2.bias)

            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

        self.ar_predict_layer = raw_t2s.model.ar_predict_layer
        self.max_sec = raw_t2s.config["data"]["max_sec"]
        self.top_k = int(raw_t2s.config["inference"]["top_k"])
        self.early_stop_num = torch.LongTensor([self.hz * self.max_sec])

    def forward(
        self,
        prompts: LongTensor,
        ref_seq: LongTensor,
        text_seq: LongTensor,
        ref_bert: torch.Tensor,
        text_bert: torch.Tensor,
        top_k: LongTensor,
    ):
        bert = torch.cat([ref_bert.T, text_bert.T], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)

        x = self.ar_text_embedding(all_phoneme_ids)

        # dtype 불일치 방지
        bert = bert.to(dtype=self.bert_proj.weight.dtype)

        x = x + self.bert_proj(bert.transpose(1, 2))
        x: torch.Tensor = self.ar_text_position(x)

        early_stop_num = self.early_stop_num

        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )

        idx = 0
        top_k = int(top_k)

        xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)

        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits = logits[:, :-1]
        samples = sample(logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]
        y = torch.concat([y, samples], dim=1)
        y_emb = self.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
            :, y_len + idx
        ].to(dtype=y_emb.dtype, device=y_emb.device)

        stop = False
        for idx in range(1, 1500):
            xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx < 11:  # 최소 10 토큰은 생성 (0.4s)
                logits = logits[:, :-1]

            samples = sample(logits, y, top_k=top_k, top_p=1, repetition_penalty=1.35, temperature=1.0)[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)
