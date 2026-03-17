"""파티션 병합 + SingleGPU BucketSampler."""
from __future__ import annotations

import glob
import os

import torch
from loguru import logger
from torch.utils.data import Sampler


def merge_partitioned_files(opt_dir: str) -> None:
    """파티셔닝된 전처리 파일을 병합한다.

    name2text-*.txt → name2text.txt
    name2semantic-*.tsv → name2semantic.tsv
    """
    for pattern, merged_name in [
        ("name2text-*.txt", "name2text.txt"),
        ("name2semantic-*.tsv", "name2semantic.tsv"),
    ]:
        merged_path = os.path.join(opt_dir, merged_name)
        if os.path.exists(merged_path):
            continue
        parts = sorted(glob.glob(os.path.join(opt_dir, pattern)))
        if not parts:
            continue
        with open(merged_path, "w", encoding="utf8") as out:
            for part in parts:
                with open(part, "r", encoding="utf8") as f:
                    content = f.read()
                    if content and not content.endswith("\n"):
                        content += "\n"
                    out.write(content)
        logger.info("{} 파일 → {}", len(parts), merged_path)


class SingleGPUBucketSampler(Sampler):
    """싱글 GPU 전용 BucketSampler.

    배치 내 시퀀스 길이를 유사하게 유지하여 패딩을 최소화한다.
    DistributedBucketSampler에서 DDP(num_replicas/rank) 로직을 제거한 버전.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        boundaries: list[int],
        shuffle: bool = True,
    ):
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.epoch = 0

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            rem = (self.batch_size - (len_bucket % self.batch_size)) % self.batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        return -1

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """에포크를 설정한다 (셔플 시드 변경)."""
        self.epoch = epoch
