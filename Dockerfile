# GPT-SoVITS-ko TTS Service
#
# 단일 Dockerfile 에서 두 변종을 빌드한다 — multi-stage --target 분기:
#
#   docker build --target infer -t gpt-sovits-ko:infer .   # 추론 전용 (가벼움)
#   docker build --target train -t gpt-sovits-ko:train .   # 학습 + 추론
#
# 기본 target 은 infer (--target 생략 시 추론 이미지 빌드).
#
# 두 변종의 차이:
# - infer: CUDA base 베이스 + dependencies 만 + 추론 소스 (scripts/inference, tools/audio,
#   tools/AP_BWE_main, tools/utils) — 학습 코드는 들어가지 않는다. TTS_MODE=infer.
# - train: CUDA cudnn-runtime 베이스 + [training,voice-checker] extras + 전체 소스.
#   ctranslate2 / faster-whisper 가 시스템 cuDNN 을 dlopen 하므로 cudnn-runtime 필수.
#
# 컨테이너 내부 경로는 호스트 monorepo 경로와 동일하게 둔다 (다른 백엔드 서비스와 일관).
# bind mount 시 호스트와 컨테이너 경로가 1:1 로 매칭되어 디버깅이 직관적이다.
#
# 모델 / voice 데이터 / 설정 / 로그는 모두 compose 의 bind mount 로만 제공 — 이미지는 stateless.
# 의존성은 pyproject.toml 이 single source of truth.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: builder-infer — 추론용 의존성 + 소스
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS builder-infer

WORKDIR /app/module-services/tts-service/GPT-SoVITS-ko

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 추론에 필요한 소스만 — 학습 전용 디렉토리는 의도적으로 제외.
COPY pyproject.toml ./
COPY main.py _setup_paths.py config.yaml ./
COPY src/                         src/
COPY scripts/_bootstrap.py        scripts/_bootstrap.py
COPY scripts/inference/           scripts/inference/

# 벤더 코드 — pretrained_models 는 볼륨 주입이라 제외.
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

# tools — 추론 런타임에 필요한 것만:
#   tools/audio              : TTS_infer_pack/TTS.py 의 v3/v4 super_res
#   tools/AP_BWE_main        : tools/audio/super_res.py 가 sys.path 로 import
#   tools/label-review.html  : /review 라우터가 서빙
#   tools/utils              : scripts/inference 의 load_audio / clean_path (fallback,
#                              사용자가 컨테이너 내에서 단독 inference 스크립트 호출 시)
COPY tools/audio/        tools/audio/
COPY tools/AP_BWE_main/  tools/AP_BWE_main/
COPY tools/label-review.html tools/label-review.html
COPY tools/utils/        tools/utils/

# 의존성 + 프로젝트 설치 (extras 없음). BuildKit cache mount 로 wheel 캐시 영속화.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/bin/python3.12 \
    && uv pip install --python /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python .

RUN /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python -m nltk.downloader \
        -d /app/module-services/tts-service/GPT-SoVITS-ko/.venv/nltk_data \
        averaged_perceptron_tagger_eng \
        cmudict \
        punkt_tab


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: builder-train — 학습 의존성 + 전체 소스
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS builder-train

WORKDIR /app/module-services/tts-service/GPT-SoVITS-ko

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 학습 이미지는 전체 소스가 필요 (scripts/training / tools/uvr5 / tools/asr / tools/voice-checker / ...).
COPY pyproject.toml ./
COPY main.py _setup_paths.py config.yaml ./
COPY src/        src/
COPY scripts/    scripts/
COPY tools/      tools/

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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/bin/python3.12 \
    && uv pip install --python /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python \
        ".[training,voice-checker]"

RUN /app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin/python -m nltk.downloader \
        -d /app/module-services/tts-service/GPT-SoVITS-ko/.venv/nltk_data \
        averaged_perceptron_tagger_eng \
        cmudict \
        punkt_tab


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a: train — 학습 런타임
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS train

WORKDIR /app/module-services/tts-service/GPT-SoVITS-ko

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        ffmpeg libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-train /app/module-services/tts-service/GPT-SoVITS-ko \
                          /app/module-services/tts-service/GPT-SoVITS-ko

RUN mkdir -p /app/module-services/tts-service/GPT-SoVITS-ko/logs \
             /app/module-services/tts-service/GPT-SoVITS-ko/data

ENV PATH="/app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV TTS_MODE=train

EXPOSE 14983

CMD ["python", "main.py", "serve"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b: infer — 추론 런타임 (기본 target — --target 생략 시 이게 빌드됨)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS infer

WORKDIR /app/module-services/tts-service/GPT-SoVITS-ko

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        ffmpeg libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-infer /app/module-services/tts-service/GPT-SoVITS-ko \
                          /app/module-services/tts-service/GPT-SoVITS-ko

RUN mkdir -p /app/module-services/tts-service/GPT-SoVITS-ko/logs \
             /app/module-services/tts-service/GPT-SoVITS-ko/data

ENV PATH="/app/module-services/tts-service/GPT-SoVITS-ko/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV TTS_MODE=infer

EXPOSE 14983

CMD ["python", "main.py", "serve"]
