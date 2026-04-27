# GPT-SoVITS-ko TTS Service
#
# GPU 추론을 수행하므로 CUDA 런타임 베이스 이미지를 사용한다.
# CUDA 버전은 compose의 build-arg로 주입 → 호스트 드라이버에 맞춰 조정 가능.
# 모델/음성 데이터/설정/로그는 모두 compose의 bind mount로만 제공한다 (이미지는 stateless).
#
# 컨테이너 내부 경로는 호스트 monorepo 경로와 동일하게 둔다 (다른 백엔드 서비스와 일관):
#   /app/module-services/tts-service/GPT-SoVITS-ko
# bind mount 시 호스트와 컨테이너 경로가 1:1 로 매칭되어 디버깅이 직관적이다.
#
# 의존성 관리: pyproject.toml 을 single source of truth 로 사용.
# - `uv pip install .` 로 프로젝트 자기 자신 + 모든 외부 의존성 설치
# - `[tool.uv.index]` / `[tool.uv.sources]` 가 그대로 적용되어 pytorch-cu126 채널로 토치 수급
# - Dockerfile 에 별도 하드코드된 패키지 목록 없음 — 의존성 추가는 pyproject.toml 만 수정

ARG CUDA_VERSION=12.6.3
ARG CUDNN_VARIANT=cudnn
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_VARIANT}-runtime-ubuntu${UBUNTU_VERSION}

ARG USER_UID=1000
ARG USER_GID=1000

WORKDIR /app/module-services/tts-service/GPT-SoVITS-ko

# 시스템 패키지 — Python 3.12 + 오디오 처리(ffmpeg/libsndfile) + 빌드 도구(일부 wheel이 source build)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        ffmpeg libsndfile1 \
        build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 서비스 소스 (uv pip install . 가 wheel 빌드하려면 src 가 있어야 한다).
# pyproject.toml 의 dependencies 가 외부 패키지 단일 source of truth.
COPY pyproject.toml ./
COPY main.py _setup_paths.py config.yaml ./
COPY src/     src/
COPY scripts/ scripts/
COPY tools/   tools/

# 벤더 코드 (GPT_SoVITS) — pretrained_models 는 볼륨으로 주입되므로 제외
COPY GPT_SoVITS/AR                GPT_SoVITS/AR/
COPY GPT_SoVITS/BigVGAN           GPT_SoVITS/BigVGAN/
COPY GPT_SoVITS/TTS_infer_pack    GPT_SoVITS/TTS_infer_pack/
COPY GPT_SoVITS/configs           GPT_SoVITS/configs/
COPY GPT_SoVITS/eres2net          GPT_SoVITS/eres2net/
COPY GPT_SoVITS/f5_tts            GPT_SoVITS/f5_tts/
COPY GPT_SoVITS/feature_extractor GPT_SoVITS/feature_extractor/
COPY GPT_SoVITS/module            GPT_SoVITS/module/
COPY GPT_SoVITS/text              GPT_SoVITS/text/
COPY GPT_SoVITS/*.py              GPT_SoVITS/

# 의존성 + 프로젝트 자체 설치. Python 경로를 명시하여 uv 가 임의 Python 을
# 다운로드하지 않도록 한다 (비-root 실행 시 /root/.local 접근 불가 문제 회피).
RUN uv venv --python /usr/bin/python3.12 \
    && uv pip install --python /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python . \
    && rm -rf /root/.cache

# NLTK 영어 G2P 리소스 — en_G2p 에서 pos_tag/word_tokenize 호출 시 필요.
# 누락 시 한영 혼합 또는 text_lang=en 요청이 LookupError 로 실패한다.
RUN /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python -m nltk.downloader \
        -d /app/module-services/tts-service/GPT-SoVITS-ko/.venv/nltk_data \
        averaged_perceptron_tagger_eng \
        cmudict \
        punkt_tab

# 런타임 사용자 — compose build-arg로 호스트 UID/GID를 주입받아 bind mount 파일 소유권과 일치시킨다.
# ubuntu 24.04 base 이미지는 기본 `ubuntu:ubuntu`(1000:1000) 사용자를 포함하므로 먼저 제거한다.
# logs, data 는 bind mount로 덮이기 전에도 app 소유여야 loguru의 mkdir(exist_ok=True) 가
# 성공한다(볼륨 누락 시 fallback). GPT_SoVITS/pretrained_models 는 항상 volume이라 제외.
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupdel ubuntu 2>/dev/null || true; \
    groupadd -g ${USER_GID} app \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/sh app \
    && mkdir -p /app/module-services/tts-service/GPT-SoVITS-ko/logs \
                /app/module-services/tts-service/GPT-SoVITS-ko/data \
    && chown -R ${USER_UID}:${USER_GID} \
                /app/module-services/tts-service/GPT-SoVITS-ko/logs \
                /app/module-services/tts-service/GPT-SoVITS-ko/data

ENV HOME=/home/app
ENV PATH="/app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 14983

USER app

CMD ["python", "main.py", "serve"]
