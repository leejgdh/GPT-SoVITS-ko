import ctypes
import os
import sys
from pathlib import Path

import ffmpeg
import numpy as np
import pandas as pd
from loguru import logger


def load_audio(file, sr):
    try:
        file = clean_path(file)
        if os.path.exists(file) is False:
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True)
        )
        raise RuntimeError("오디오 로드 실패")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str: str):
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(" '\n\"\u202a")


def check_for_existance(file_list: list = None, is_train=False, is_dataset_processing=False):
    files_status = []
    if is_train and file_list:
        file_list.append(os.path.join(file_list[0], "2-name2text.txt"))
        file_list.append(os.path.join(file_list[0], "3-bert"))
        file_list.append(os.path.join(file_list[0], "4-cnhubert"))
        file_list.append(os.path.join(file_list[0], "5-wav32k"))
        file_list.append(os.path.join(file_list[0], "6-name2semantic.tsv"))
    for file in file_list:
        files_status.append(os.path.exists(file))
    if sum(files_status) != len(files_status):
        missing = [f for f, ok in zip(file_list, files_status) if not ok]
        if is_train:
            for f in missing:
                logger.warning("파일/폴더 없음: {}", f)
            return False
        elif is_dataset_processing:
            if files_status[0]:
                return True
            logger.warning("파일/폴더 없음: {}", file_list[0])
            return False
        else:
            if file_list[0]:
                logger.warning("파일/폴더 없음: {}", file_list[0])
            else:
                logger.warning("경로가 비어 있습니다")
            return False
    return True


def check_details(path_list=None, is_train=False, is_dataset_processing=False):
    if is_dataset_processing:
        list_path, audio_path = path_list
        if not list_path.endswith(".list"):
            logger.warning("올바른 .list 경로를 입력하세요")
            return
        if audio_path:
            if not os.path.isdir(audio_path):
                logger.warning("올바른 오디오 폴더 경로를 입력하세요")
                return
        with open(list_path, "r", encoding="utf8") as f:
            line = f.readline().strip("\n").split("\n")
        wav_name, _, __, ___ = line[0].split("|")
        wav_name = clean_path(wav_name)
        if audio_path:
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s" % (audio_path, wav_name)
        else:
            wav_path = wav_name
        if not os.path.exists(wav_path):
            logger.warning("경로 오류: {}", wav_path)
        return
    if is_train:
        path_list.append(os.path.join(path_list[0], "2-name2text.txt"))
        path_list.append(os.path.join(path_list[0], "4-cnhubert"))
        path_list.append(os.path.join(path_list[0], "5-wav32k"))
        path_list.append(os.path.join(path_list[0], "6-name2semantic.tsv"))
        phone_path, hubert_path, wav_path, semantic_path = path_list[1:]
        with open(phone_path, "r", encoding="utf-8") as f:
            if not f.read(1):
                logger.warning("음소 데이터셋이 비어 있습니다")
        if not os.listdir(hubert_path):
            logger.warning("HuBERT 데이터셋이 비어 있습니다")
        if not os.listdir(wav_path):
            logger.warning("오디오 데이터셋이 비어 있습니다")
        df = pd.read_csv(semantic_path, delimiter="\t", encoding="utf-8")
        if len(df) < 1:
            logger.warning("시맨틱 데이터셋이 비어 있습니다")


def load_cudnn():
    import torch

    if not torch.cuda.is_available():
        logger.info("CUDA is not available, skipping cuDNN setup.")
        return

    if sys.platform == "win32":
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.exists():
            os.add_dll_directory(str(torch_lib_dir))
            logger.info("Added DLL directory: {}", torch_lib_dir)
            matching_files = sorted(torch_lib_dir.glob("cudnn_cnn*.dll"))
            if not matching_files:
                logger.error("No cudnn_cnn*.dll found in {}", torch_lib_dir)
                return
            for dll_path in matching_files:
                dll_name = os.path.basename(dll_path)
                try:
                    ctypes.CDLL(dll_name)
                    logger.info("Loaded: {}", dll_name)
                except OSError as e:
                    logger.warning("Failed to load {}: {}", dll_name, e)
        else:
            logger.warning("Torch lib directory not found: {}", torch_lib_dir)

    elif sys.platform == "linux":
        site_packages = Path(torch.__file__).resolve().parents[1]
        cudnn_dir = site_packages / "nvidia" / "cudnn" / "lib"

        if not cudnn_dir.exists():
            logger.error("cudnn dir not found: {}", cudnn_dir)
            return

        matching_files = sorted(cudnn_dir.glob("libcudnn_cnn*.so*"))
        if not matching_files:
            logger.error("No libcudnn_cnn*.so* found in {}", cudnn_dir)
            return

        for so_path in matching_files:
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)  # type: ignore
                logger.info("Loaded: {}", so_path)
            except OSError as e:
                logger.warning("Failed to load {}: {}", so_path, e)


def load_nvrtc():
    import torch

    if not torch.cuda.is_available():
        logger.info("CUDA is not available, skipping nvrtc setup.")
        return

    if sys.platform == "win32":
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.exists():
            os.add_dll_directory(str(torch_lib_dir))
            logger.info("Added DLL directory: {}", torch_lib_dir)
            matching_files = sorted(torch_lib_dir.glob("nvrtc*.dll"))
            if not matching_files:
                logger.error("No nvrtc*.dll found in {}", torch_lib_dir)
                return
            for dll_path in matching_files:
                dll_name = os.path.basename(dll_path)
                try:
                    ctypes.CDLL(dll_name)
                    logger.info("Loaded: {}", dll_name)
                except OSError as e:
                    logger.warning("Failed to load {}: {}", dll_name, e)
        else:
            logger.warning("Torch lib directory not found: {}", torch_lib_dir)

    elif sys.platform == "linux":
        site_packages = Path(torch.__file__).resolve().parents[1]
        nvrtc_dir = site_packages / "nvidia" / "cuda_nvrtc" / "lib"

        if not nvrtc_dir.exists():
            logger.error("nvrtc dir not found: {}", nvrtc_dir)
            return

        matching_files = sorted(nvrtc_dir.glob("libnvrtc*.so*"))
        if not matching_files:
            logger.error("No libnvrtc*.so* found in {}", nvrtc_dir)
            return

        for so_path in matching_files:
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)  # type: ignore
                logger.info("Loaded: {}", so_path)
            except OSError as e:
                logger.warning("Failed to load {}: {}", so_path, e)
