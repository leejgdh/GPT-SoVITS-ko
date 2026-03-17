"""체크포인트 버전 감지 + 로드 유틸리티.

추론/내보내기에서 사용하는 체크포인트 관련 함수를 제공한다.
학습용 체크포인트 저장/로드는 tools/training/checkpoint.py를 참조.
"""
import hashlib
import os
from io import BytesIO

import torch

# 파일 선두 2바이트 → [text_model_version, sovits_version, is_lora]
head2version = {
    b"00": ["v1", "v1", False],
    b"01": ["v2", "v2", False],
    b"02": ["v2", "v3", False],
    b"03": ["v2", "v3", True],
    b"04": ["v2", "v4", True],
    b"05": ["v2", "v2Pro", False],
    b"06": ["v2", "v2ProPlus", False],
}
hash_pretrained_dict = {
    "dc3c97e17592963677a4a1681f30c653": ["v2", "v2", False],  # s2G488k.pth#sovits_v1_pretrained
    "43797be674a37c1c83ee81081941ed0f": ["v2", "v3", False],  # s2Gv3.pth#sovits_v3_pretrained
    "6642b37f3dbb1f76882b69937c95a5f3": ["v2", "v2", False],  # s2G2333K.pth#sovits_v2_pretrained
    "4f26b9476d0c5033e04162c486074374": ["v2", "v4", False],  # s2Gv4.pth#sovits_v4_pretrained
    "c7e9fce2223f3db685cdfa1e6368728a": ["v2", "v2Pro", False],  # s2Gv2Pro.pth#sovits_v2Pro_pretrained
    "66b313e39455b57ab1b0bc0b239c9d0a": ["v2", "v2ProPlus", False],  # s2Gv2ProPlus.pth#sovits_v2ProPlus_pretrained
}


def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def get_sovits_version_from_path_fast(sovits_path):
    ###1-if it is pretrained sovits models, by hash
    hash = get_hash_from_file(sovits_path)
    if hash in hash_pretrained_dict:
        return hash_pretrained_dict[hash]
    ###2-new weights, by head
    with open(sovits_path, "rb") as f:
        version = f.read(2)
    if version != b"PK":
        return head2version[version]
    ###3-old weights, by file size
    if_lora_v3 = False
    size = os.path.getsize(sovits_path)
    """
            v1weights:about 82942KB
                half thr:82978KB
            v2weights:about 83014KB
            v3weights:about 750MB
    """
    if size < 82978 * 1024:
        model_version = version = "v1"
    elif size < 700 * 1024 * 1024:
        model_version = version = "v2"
    else:
        version = "v2"
        model_version = "v3"
    return version, model_version, if_lora_v3


def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)
