# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/dataset.py
# reference: https://github.com/lifeiteng/vall-e

import os
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

version = os.environ.get("version", None)

from text import cleaned_text_to_sequence


def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        phoneme_path: str,
        semantic_path: str,
        max_sample: int = None,
        max_sec: int = 100,
        pad_val: int = 1024,
        min_ps_ratio: int = 3,
        max_ps_ratio: int = 25,
    ) -> None:
        super().__init__()

        self.semantic_data = pd.read_csv(
            semantic_path,
            delimiter="\t",
            encoding="utf-8",
        )
        self.path2 = phoneme_path
        self.path6 = semantic_path
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path6)
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1], tmp[2], tmp[3]]

        self.PAD: int = pad_val
        self.hz = int(os.environ.get("hz", "25hz")[:-2])

        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        if max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]

        self.semantic_phoneme = []
        self.item_names = []

        self.init_batch()
        del self.semantic_data
        del self.phoneme_data

    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        logger.info("semantic_data_len: {}, phoneme_data_len: {}", semantic_data_len, phoneme_data_len)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(semantic_data_len):
            item_name = self.semantic_data.iloc[i, 0]
            try:
                phoneme, word2ph, text = self.phoneme_data[item_name]
            except KeyError:
                num_not_in += 1
                continue

            semantic_str = self.semantic_data.iloc[i, 1]
            semantic_ids = [int(idx) for idx in semantic_str.split(" ")]
            # max_sec 기준으로 긴 샘플 필터링
            if len(semantic_ids) > self.max_sec * self.hz:
                num_deleted_bigger += 1
                continue
            phoneme = phoneme.split(" ")

            try:
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except Exception:
                traceback.print_exc()
                num_not_in += 1
                continue
            if len(phoneme_ids) > self.max_sec * self.hz / 2.5:
                num_deleted_ps += 1
                continue

            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)

            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                num_deleted_ps += 1
                continue

            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            idx += 1
            self.item_names.append(item_name)

        # 데이터가 너무 적으면 복제하여 최소 수량 확보
        min_num = 100
        leng = len(self.semantic_phoneme)
        if leng < min_num:
            tmp1 = self.semantic_phoneme
            tmp2 = self.item_names
            self.semantic_phoneme = []
            self.item_names = []
            for _ in range(max(2, int(min_num / leng))):
                self.semantic_phoneme += tmp1
                self.item_names += tmp2
        if num_not_in > 0:
            logger.warning("시맨틱 데이터 중 음소 데이터에 없는 항목: {}개", num_not_in)
        if num_deleted_bigger > 0:
            logger.info("{}초 초과 오디오 필터링: {}개", self.max_sec, num_deleted_bigger)
        if num_deleted_ps > 0:
            logger.info(
                "phoneme/sec 범위({}-{}) 밖 오디오 필터링: {}개",
                self.min_ps_ratio, self.max_ps_ratio, num_deleted_ps,
            )
        logger.info("dataset.__len__(): {}", self.__len__())

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        phoneme_ids_len = len(phoneme_ids)
        semantic_ids_len = len(semantic_ids)

        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_len,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)

        # BERT 특징은 중국어 전용이므로 제로 텐서로 대체
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        return {
            "ids": sample_index,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_lens,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_lens,
            "bert_feature": bert_padded,
        }
