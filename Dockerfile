# GPT-SoVITS-ko TTS Service
#
# 빌드 컨텍스트: 프로젝트 루트 (GPT-SoVITS-ko/)
#
# 볼륨 마운트 필요:
#   - data/voice/              학습 데이터 & voice.yaml
#   - GPT_SoVITS/pretrained_models/  사전 학습 모델 (2.2GB+)
#   - conf.yaml                서비스 설정
#
# 빌드:
#   docker build -t gpt-sovits-ko .
#
# 실행:
#   docker run --gpus all -p 14983:14983 \
#     -v ./data/voice:/app/data/voice \
#     -v ./GPT_SoVITS/pretrained_models:/app/GPT_SoVITS/pretrained_models \
#     -v ./conf.yaml:/app/conf.yaml:ro \
#     gpt-sovits-ko

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        ffmpeg libsndfile1 \
        build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 설치 (소스 변경 시 캐시 활용)
COPY pyproject.toml uv.lock ./
RUN uv venv && \
    uv sync --frozen --no-dev --no-install-project && \
    rm -rf /root/.cache

# 벤더 코드 (GPT_SoVITS)
COPY GPT_SoVITS/AR            GPT_SoVITS/AR/
COPY GPT_SoVITS/BigVGAN       GPT_SoVITS/BigVGAN/
COPY GPT_SoVITS/TTS_infer_pack GPT_SoVITS/TTS_infer_pack/
COPY GPT_SoVITS/configs       GPT_SoVITS/configs/
COPY GPT_SoVITS/eres2net      GPT_SoVITS/eres2net/
COPY GPT_SoVITS/f5_tts        GPT_SoVITS/f5_tts/
COPY GPT_SoVITS/feature_extractor GPT_SoVITS/feature_extractor/
COPY GPT_SoVITS/module        GPT_SoVITS/module/
COPY GPT_SoVITS/text          GPT_SoVITS/text/
COPY GPT_SoVITS/*.py          GPT_SoVITS/

# 소스 코드
COPY src/     src/
COPY scripts/ scripts/
COPY tools/   tools/
COPY main.py _setup_paths.py ./
COPY conf.example.yaml conf.yaml

# pretrained_models, data/voice 는 볼륨 마운트
RUN mkdir -p GPT_SoVITS/pretrained_models data/voice data/voice-checker logs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 14983

CMD ["python", "main.py", "serve"]
