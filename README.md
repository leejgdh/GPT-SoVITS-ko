# GPT-SoVITS-ko

한국어 특화 음성 복제 TTS 서비스. [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT) 를 기반으로 WebUI 를 제거하고 **CLI 파이프라인 + REST API** 구조로 재설계했습니다.

> 소량의 음성 데이터 (1~5분) 로 화자 목소리를 학습하고, REST API 로 실시간 합성.

---

## 핵심 기능

- **단일 명령 E2E 파이프라인**: raw audio → 전처리 → 학습 → voice.yaml 생성까지 한 번에
- **자동 모델 다운로드**: pretrained weight / Whisper / UVR5 / FRCRN 등 첫 호출 시 자동 수급 — 사전 준비 불필요
- **학습 / 추론 이미지 분리**: Docker `--target infer` (가벼움) vs `--target train` (학습 의존성 포함)
- **REST API + 검수 UI**: FastAPI 기반 합성 + 학습 중간 산출물 (ASR 라벨) 검수
- **학습 산출물 자동 정리**: `cleanup-voice` 명령으로 voice.yaml 의 weight/ref_audio 만 남기고 회수

---

## 디렉토리

```
GPT-SoVITS-ko/
├── main.py                 # CLI 엔트리포인트
├── config.yaml             # 서비스 설정
├── src/
│   ├── server/             # FastAPI 라우터
│   ├── cli/                # serve / pipeline / cleanup 등 명령 dispatcher
│   └── config/             # config / voice profile dataclass
├── scripts/
│   ├── data_preparation/   # denoise, slice, uvr5, asr
│   ├── preprocessing/      # text, hubert, sv, semantic
│   ├── training/           # s1 (GPT) / s2 (SoVITS)
│   └── inference/          # inference_cli (step4)
├── GPT_SoVITS/             # 원본 모델 코드 (RVC-Boss 베이스)
│   └── pretrained_models/  # 베이스 weight (자동 다운로드, gitignored)
├── tools/                  # uvr5, asr, voice-checker, super_res, ...
└── data/
    ├── voice/{name}/       # 화자별 학습/추론 산출물 (bind mount 대상)
    └── models/             # 다운로드된 ASR 모델 등
```

---

## 요구 환경

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| CUDA | 12.6+ (호스트 NVIDIA driver) |
| GPU VRAM | 추론 6GB+ / 학습 12GB+ 권장 |
| 패키지 관리 | [uv](https://docs.astral.sh/uv/) |

---

## E2E 워크플로우

raw 오디오 → 학습 → 합성까지 한 흐름. 화자별 voice 디렉토리 `data/voice/{agent_id}/` 가 단위 — 디렉토리 이름이 곧 agent 식별자이며 API 의 `voice` 파라미터, `voice.yaml` 의 `name` 과 일치합니다.

### 1. 학습 의존성 설치

```bash
git clone <repo>
cd GPT-SoVITS-ko
uv sync --extra training --extra voice-checker
```

> `uv sync` 만 하면 추론 의존성만 들어갑니다. 학습 / 데이터 준비 / 검수까지 하려면 `training`, `voice-checker` extras 필수.
>
> uv workspace member 로 통합된 환경에서는 워크스페이스 루트에서 `uv sync --package gpt-sovits --extra training --extra voice-checker` — 서브 디렉토리 sync 는 다른 member 를 prune 합니다.

### 2. 화자 음성 준비

```bash
mkdir -p data/voice/{agent_id}/raw_audio
cp /path/to/recordings/*.{wav,mp3,flac} data/voice/{agent_id}/raw_audio/
```

권장: 1~5분 분량, 16bit, 44.1kHz 이상, 한 화자만 깨끗하게 녹음.

### 3. 파이프라인 실행

```bash
uv run python main.py pipeline \
  --voice-dir data/voice/{agent_id} \
  --version v2Pro \
  --output-text "안녕하세요 테스트 합성입니다"
```

**첫 실행 시 자동 다운로드** (총 5~7 GB, 네트워크 따라 10~30분):

| 항목 | 출처 | 단계 |
|---|---|---|
| FRCRN denoise | modelscope | step1-denoise |
| UVR5 HP5_only_main_vocal | HuggingFace | step1-uvr5 |
| faster-whisper-large-v3 | HuggingFace | step1-asr |
| chinese-hubert-base | HuggingFace | step2-hubert |
| sv (eres2netv2) | HuggingFace | step2-sv (v2Pro/v2ProPlus) |
| s1v3.ckpt (GPT base) | HuggingFace | step3-train-gpt |
| s2G v2Pro / v2 / v3 / v4 | HuggingFace | step3-train-sovits |
| s2D discriminator | HuggingFace | step3-train-sovits |
| fast_langdetect lid.176.bin | HF (라이브러리 내장) | 추론 (언어 감지) |

→ **사전 다운로드 불필요**. 미리 받아두고 싶다면 [수동 사전 다운로드](#수동-사전-다운로드-선택) 참고.

### 4. 합성 요청

파이프라인 끝나면 `data/voice/{agent_id}/voice.yaml` 이 자동 생성됩니다. 서버 띄우고 합성:

```bash
uv run python main.py serve &
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{"voice": "{agent_id}", "text": "안녕하세요", "text_lang": "ko"}' \
  --output out.wav
```

### 5. 학습 산출물 정리 (음질 확인 후)

```bash
# 삭제 대상 미리보기
uv run python main.py cleanup-voice --voice-dir data/voice/{agent_id} --dry-run

# 실제 정리 — voice.yaml 의 weight + ref_audio 만 남기고 회수
uv run python main.py cleanup-voice --voice-dir data/voice/{agent_id}

# 옵션
#   --keep-raw  : raw_audio/ 보존 (재학습 대비)
#   --keep-asr  : step1/04_asr/ 보존 (라벨 검수 결과 재사용)
```

학습 voice 하나당 step1~3 + logs 합쳐 보통 수 GB ~ 수십 GB. 음질 OK 판정 후 회수.

---

## 사용법 — 단계별

각 step 을 개별 실행할 수 있습니다 (디버깅 / 일부 재실행):

```bash
uv run python main.py step1 --voice-dir data/voice/{agent_id}
uv run python main.py step2 --voice-dir data/voice/{agent_id} --version v2Pro
uv run python main.py step3 --voice-dir data/voice/{agent_id} --version v2Pro --epochs 8
uv run python main.py step4 --voice-dir data/voice/{agent_id} --version v2Pro --output-text "..."
```

파이프라인 흐름:

```
raw_audio/                          (사용자 입력)
  ↓ denoise   (FRCRN)
step1/01_denoise/
  ↓ slice     (무음 슬라이싱)
step1/02_sliced/
  ↓ uvr5      (보컬 분리)
step1/03_vocal/
  ↓ asr       (Whisper)             [← /review UI 로 검수]
step1/04_asr/
  ↓ get-text + get-hubert + get-sv + get-semantic
step2/{version}/                    (전처리 피처)
  ↓ s1_train (GPT AR) + s2_train (SoVITS)
step3/{version}/                    (학습 weight)
  ↓ inference_cli + voice.yaml 생성
voice.yaml (available: true)
```

전체 명령 목록: `uv run python main.py --help`

### ASR 라벨 검수 UI

step1 또는 pipeline 실행 중에 백그라운드 서버가 자동으로 뜹니다:

```
http://localhost:9880/review
```

- 오디오 재생 + 파형
- pending → approved / rejected 상태 변경
- 텍스트 인라인 편집
- 감정 매핑

기본은 `pending` 모두 학습 데이터로 사용. `approved` 가 하나라도 있으면 그것만 사용 (rejected 제외).

---

## 운영 — Docker

### 빌드 (학습 / 추론 분리)

단일 Dockerfile, multi-stage `--target` 분기:

```bash
# 추론 전용 (CUDA base, ~7 GB, training dep 없음)
docker build --target infer -t gpt-sovits-ko:infer .

# 학습 + 추론 (CUDA cudnn-runtime, ~9 GB, 전체 dep)
docker build --target train -t gpt-sovits-ko:train .
```

`--target` 생략 시 `infer` 가 기본.

### 실행

`pretrained_models` / `data` / `logs` 는 bind mount — 이미지 stateless:

```bash
docker run -d --gpus all -p 9880:14983 \
  -v $PWD/data:/app/module-services/tts-service/GPT-SoVITS-ko/data \
  -v $PWD/GPT_SoVITS/pretrained_models:/app/module-services/tts-service/GPT-SoVITS-ko/GPT_SoVITS/pretrained_models \
  -v $PWD/logs:/app/module-services/tts-service/GPT-SoVITS-ko/logs \
  gpt-sovits-ko:infer
```

### 모드 차이

| 환경변수 | 이미지 | 동작 |
|---|---|---|
| `TTS_MODE=infer` | infer | `serve` / `cleanup-voice` 만 허용. 학습 명령은 즉시 거부 |
| `TTS_MODE=train` | train | 모든 명령 허용 |

추론 컨테이너에서 학습 명령을 호출하면 친절한 안내 후 종료 (exit code 2) — extras 부재로 인한 깊은 ModuleNotFoundError 회피.

### 학습은 train 이미지에서

```bash
docker run --rm --gpus all \
  -v $PWD/data:/app/module-services/tts-service/GPT-SoVITS-ko/data \
  -v $PWD/GPT_SoVITS/pretrained_models:/app/module-services/tts-service/GPT-SoVITS-ko/GPT_SoVITS/pretrained_models \
  gpt-sovits-ko:train \
  python main.py pipeline --voice-dir data/voice/{agent_id} --version v2Pro --output-text "테스트"
```

호스트와 컨테이너의 monorepo 경로를 동일하게 맞췄기 때문에 bind mount 가 1:1.

---

## API

### `POST /tts` — 합성

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|:----:|------|
| `voice` | string | O | Voice Profile 이름 |
| `text` | string | O | 합성할 텍스트 |
| `text_lang` | string | O | `ko` / `ja` / `en` / `auto` |
| `emotion` | string | - | 감정 프리셋 (기본: `default`) |
| `media_type` | string | - | `wav` / `ogg` / `aac` / `raw` |
| `streaming_mode` | int | - | 0: 일괄 / 1: fragment / 2: 스트리밍 / 3: 고정 청크 |
| `speed_factor` | float | - | 1.0 |
| `temperature` | float | - | 1.0 |
| `top_k` | int | - | 15 |

### 그 외

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /voices` | Voice 목록 |
| `GET /voices/{name}` | Voice 상세 |
| `GET /voices/{name}/emotions` | 감정 매핑 |
| `GET /voices/{name}/labels` | ASR 라벨 + 통계 |
| `GET /review` | ASR 라벨 검수 UI |
| `GET /health` | 헬스 체크 |
| `GET /metrics` | Prometheus 메트릭 |

자세한 스키마는 서버 실행 후 `http://localhost:9880/docs` (Swagger).

---

## Voice Profile

`data/voice/{name}/voice.yaml` — step4 (또는 pipeline) 완료 시 자동 생성.

```yaml
name: {agent_id}
version: v2Pro
ref_lang: ko
gpt_weights: step3/v2Pro/02_gpt_weights/{agent_id}-e5.ckpt
sovits_weights: step3/v2Pro/04_sovits_weights/{agent_id}_e4_s152.pth
available: true
emotions:
  default:
    ref_audio: step1/03_vocal/normal_001.flac
    ref_text: "평범한 톤의 참조 텍스트"
  happy:
    ref_audio: step1/03_vocal/cheerful_003.flac
    ref_text: "기쁜 톤의 참조 텍스트"
```

감정 추가는 API (`POST /voices/{name}/emotions`) 또는 yaml 직접 편집.

---

## 지원 모델 버전

| 버전 | 아키텍처 | 특징 |
|------|---------|------|
| `v2` | VITS | 한국어 기본 |
| **`v2Pro`** | VITS + SV | v2 비용으로 v3급 유사도 (권장) |
| `v2ProPlus` | VITS + SV (확장) | v2Pro 의 더 큰 모델 |
| `v3` / `v4` | CFM / DiT | 감정 표현 우수, 더 무거움 |

`--version v2Pro` 가 가장 균형 잡힘. v3/v4 는 super_res (24k→48k) 가 함께 사용됩니다.

---

## 설정

`config.yaml` 의 주요 키 (전체는 `src/config/config.py` 참조):

```yaml
service:
  host: 0.0.0.0
  port: 14983
  voices_dir: data/voice          # voice 디렉토리 root

voice_checker:                     # 선택 — CNN 품질 분류기 (학습 후 활성화)
  inference:
    model_path: data/voice-checker/models/best_model.pth
```

환경변수:

| 변수 | 의미 |
|---|---|
| `TTS_MODE` | `infer` 면 학습 명령 거부. Docker stage 가 자동 설정 |
| `TTS_SERVICE_CONFIG` | 로드된 config.yaml 절대경로 (서브프로세스에 전달용) |

---

## 수동 사전 다운로드 (선택)

자동 다운로드를 미리 받아두고 싶거나 인터넷 없는 환경이라면, 다른 머신에서 받아서 `GPT_SoVITS/pretrained_models/` 통째 복사:

```bash
# 같은 LAN 의 학습 서버에서 (예시)
rsync -avhP --inplace \
  user@server:/path/to/GPT-SoVITS-ko/GPT_SoVITS/pretrained_models/ \
  ./GPT_SoVITS/pretrained_models/
```

원본 출처는 HuggingFace `lj1995/GPT-SoVITS` 등 — 자동 다운로드 코드 (`tools/utils/download.py`) 와 동일한 repo.

---

## 자주 막히는 곳

- **`ModuleNotFoundError: faster_whisper / peft / funasr`**: `uv sync --extra training --extra voice-checker` 가 누락됐습니다.
- **추론 컨테이너에서 학습 명령 거부됨**: 정상 동작. `gpt-sovits-ko:train` 이미지로 실행하세요.
- **첫 step1 이 매우 느림**: 자동 다운로드 (FRCRN/UVR5/Whisper) 중. 두 번째부터는 빠릅니다.
- **`v3/v4` 추론 시 super_res 에러**: `tools/audio/super_res.py` 의 sys.path 가 `tools/audio/AP_BWE_main/` 를 찾는데 실제 위치는 `tools/AP_BWE_main/`. v3/v4 사용 안 하면 무관 (알려진 이슈).
- **uv workspace 환경에서 다른 member 가 깨짐**: 서브디렉토리에서 sync 시 prune 발생. 워크스페이스 통합 환경에서는 루트에서 `--package gpt-sovits` 명시.

---

## 라이선스

MIT License.

[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT, © 2024 RVC-Boss) 기반. 원본 코드는 `GPT_SoVITS/`, `tools/audio/`, `tools/asr/`, `tools/uvr5/` 에. BigVGAN (`GPT_SoVITS/BigVGAN/`) 은 NVIDIA MIT + 하위 라이선스 (`incl_licenses/` 참조). AP-BWE (`tools/AP_BWE_main/`) 는 MIT.
