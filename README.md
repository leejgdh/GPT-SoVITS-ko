# GPT-SoVITS-ko

한국어 특화 음성 복제 TTS 서비스.
[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)(MIT) 를 기반으로, WebUI를 제거하고 **CLI 파이프라인 + REST API** 구조로 재설계했습니다.

> 소량의 음성 데이터(1~5분)로 화자의 음성을 학습하고, REST API로 실시간 합성할 수 있습니다.

---

## 원본 대비 변경 사항

- **Gradio WebUI 전체 제거** → FastAPI REST API + ServiceContext 의존성 주입
- **Voice Profile** — `voice.yaml` 기반 캐릭터 자체완결 구조 (가중치 경로, 감정 프리셋 포함)
- **감정 프리셋(EmotionRef)** — 감정별 참조 오디오 매핑, API에서 `emotion` 파라미터로 선택
- **ASR 라벨 검수 시스템** — CRUD + 상태관리(pending/approved/rejected) + 웹 UI
- **원커맨드 파이프라인** — `python main.py pipeline` 으로 데이터 준비 → 전처리 → 학습 → 추론 일괄 실행
- **라우터 5분할** — tts / labels / emotions / voices / system
- **Voice Checker** — CNN 오디오 품질 분류 도구 내장 (`tools/voice-checker/`)
- **`_setup_paths.py`** — sys.path 단일 소스 관리
- `GPT_SoVITS/` 모델 아키텍처는 프리트레인 가중치 호환성을 위해 원본 유지

---

## 요구 환경

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| CUDA | 12.6+ |
| GPU VRAM | 6GB+ (학습 시 12GB+ 권장) |
| 패키지 관리 | [uv](https://docs.astral.sh/uv/) |

---

## 설치

```bash
git clone https://github.com/leejgdh/GPT-SoVITS-ko.git
cd GPT-SoVITS-ko

# 전체 의존성 설치 (서버 + 파이프라인 + Voice Checker)
uv sync

# 서버 + 추론만 (파이프라인/학습 의존성 제외, 경량 설치)
uv sync --only-group serve
```

**설정 파일 생성:**

```bash
cp conf.example.yaml conf.yaml
```

---

## 빠른 시작

### 1. 서버 실행

```bash
uv run python main.py serve
uv run python main.py serve -c conf.yaml --port 8080
```

### 2. 원커맨드 파이프라인

원본 오디오만 있으면 데이터 준비 → 전처리 → 학습 → 추론을 한 번에 실행합니다.

```bash
# data/voice/dahwi/raw_audio/ 에 음성 파일(WAV, 16bit, 44.1kHz+)을 넣고:
uv run python main.py pipeline \
  --voice-dir data/voice/dahwi \
  --output-text "합성할 텍스트" \
  --version v2Pro
```

완료 시 `voice.yaml`이 자동 생성되고, 서버에서 바로 사용할 수 있습니다.

### 3. TTS 합성 요청

```bash
curl -X POST http://localhost:9880/tts \
  -H "Content-Type: application/json" \
  -d '{"voice": "dahwi", "text": "안녕하세요", "text_lang": "ko", "emotion": "default"}' \
  --output output.wav
```

---

## 파이프라인 상세

4단계 파이프라인으로 구성되며, 각 단계를 개별 실행할 수도 있습니다.

```
[원본 오디오]
    │
    ├─ Step 1: 데이터 준비
    │   ├─ FRCRN 노이즈 제거
    │   ├─ 무음 기반 슬라이싱
    │   ├─ UVR5 보컬 분리
    │   ├─ Whisper ASR (음성→텍스트)
    │   └─ Voice Checker CNN 품질 분류 (선택)
    │
    ├─ Step 2: 전처리
    │   ├─ 음소 추출 (다국어)
    │   ├─ HuBERT + wav32k 변환
    │   ├─ 화자 임베딩 (v2Pro/v2ProPlus)
    │   └─ Semantic 토큰 추출
    │
    ├─ Step 3: 학습
    │   ├─ GPT AR 모델 (Text-to-Semantic)
    │   └─ SoVITS Vocoder (VITS / CFM)
    │
    └─ Step 4: 추론
        ├─ 테스트 합성
        └─ voice.yaml 자동 생성
```

### 개별 스텝 실행

각 step을 독립적으로 실행할 수 있습니다.

```bash
# Step 1: 데이터 준비 (denoise → slice → UVR5 → ASR)
uv run python main.py step1 --voice-dir data/voice/dahwi

# Step 2: 전처리 (text → hubert → semantic)
uv run python main.py step2 --voice-dir data/voice/dahwi --version v2Pro

# Step 3: 학습 (GPT AR + SoVITS)
uv run python main.py step3 --voice-dir data/voice/dahwi --version v2Pro
uv run python main.py step3 --voice-dir data/voice/dahwi --version v2Pro --epochs 20

# Step 4: 추론 + voice.yaml 생성
uv run python main.py step4 --voice-dir data/voice/dahwi --version v2Pro \
  --output-text "합성할 텍스트"
```

---

## API

### TTS 합성

```
POST /tts
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|:----:|------|
| `voice` | string | O | Voice Profile 이름 |
| `text` | string | O | 합성할 텍스트 |
| `text_lang` | string | O | 텍스트 언어 (`ko`, `ja`, `en`, `zh`, `auto`) |
| `emotion` | string | - | 감정 프리셋 (기본: `default`) |
| `media_type` | string | - | 출력 포맷 (`wav`, `ogg`, `aac`, `raw`) |
| `streaming_mode` | int | - | 0: 일괄, 1: fragment, 2: 스트리밍, 3: 고정 청크 |
| `speed_factor` | float | - | 속도 조절 (기본: 1.0) |
| `volume` | float | - | 볼륨 게인 (기본: 1.0) |
| `temperature` | float | - | 샘플링 온도 (기본: 1.0) |
| `top_k` | int | - | Top-K 샘플링 (기본: 15) |

### Voice 관리

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/voices` | GET | voice 프로필 목록 + 현재 로드된 voice |
| `/voices/{name}` | GET | voice 상세 (version, ref_lang, emotions) |

### ASR 라벨 관리

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/voices/{name}/labels` | GET | 라벨 목록 + 상태별 통계 |
| `/voices/{name}/labels/{index}/audio` | GET | 라벨 오디오 스트리밍 |
| `/voices/{name}/labels/{index}` | PUT | 라벨 텍스트/언어 수정 |
| `/voices/{name}/labels/{index}/state` | PATCH | 상태 변경 (pending/approved/rejected) |
| `/voices/{name}/labels/{index}` | DELETE | 라벨 + 오디오 삭제 |

### 감정 매핑

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/voices/{name}/emotions` | GET | 감정 목록 |
| `/voices/{name}/emotions/{emotion}` | PUT | 라벨 인덱스 → 감정 매핑 |
| `/voices/{name}/emotions/{emotion}` | DELETE | 감정 삭제 |

### 시스템

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 헬스 체크 |
| `/review` | GET | ASR 라벨 검수 UI |
| `/control` | GET | 서버 제어 (`?command=restart\|exit`) |

---

## Voice Profile

각 캐릭터의 음성 설정은 `data/voice/{name}/voice.yaml`로 관리합니다.
파이프라인 Step 4 완료 시 자동 생성되며, 감정 매핑은 API를 통해 추가합니다.

```yaml
name: dahwi
version: v2Pro
ref_lang: ko
gpt_weights: step3/v2Pro/02_gpt_weights/dahwi-e5.ckpt
sovits_weights: step3/v2Pro/04_sovits_weights/dahwi_e4_s152.pth
emotions:
  default:
    ref_audio: step1/03_vocal/normal_001.flac
    ref_text: "평범한 톤의 참조 텍스트"
  happy:
    ref_audio: step1/03_vocal/happy_003.flac
    ref_text: "기쁜 톤의 참조 텍스트"
```

---

## 데이터 디렉토리 구조

```
data/voice/{character}/
├── raw_audio/                # 원본 음성 파일 (WAV, 16bit, 44.1kHz+)
├── step1/                    # 데이터 준비
│   ├── 01_denoise/           #   FRCRN 노이즈 제거
│   ├── 02_sliced/            #   무음 기반 슬라이싱
│   ├── 03_vocal/             #   UVR5 보컬 분리
│   └── 04_asr/               #   Whisper ASR 라벨 (.list)
├── step2/{version}/          # 전처리 (음소, HuBERT, semantic)
├── step3/{version}/          # 학습
│   ├── 01_gpt_logs/          #   GPT 학습 로그 + 체크포인트
│   ├── 02_gpt_weights/       #   GPT half-precision 가중치
│   ├── 03_sovits_logs/       #   SoVITS 학습 로그 + 체크포인트
│   └── 04_sovits_weights/    #   SoVITS half-precision 가중치
├── step4/{version}/          # 추론 출력
└── voice.yaml                # Voice Profile (자동 생성)
```

---

## 도구

### ASR 라벨 검수 UI

서버 실행 후 `/review` 엔드포인트로 접속합니다.
Whisper ASR이 생성한 라벨을 검수하고 감정 매핑까지 수행하는 웹 UI입니다.

- 라벨 상태 관리: pending → approved / rejected
- 오디오 재생 + 파형 시각화
- 텍스트 인라인 편집
- 감정 프리셋 매핑
- 빈 텍스트 라벨 일괄 삭제

```bash
uv run python main.py serve
# → http://localhost:9880/review
```

### Voice Checker

`tools/voice-checker/` — 오디오 품질을 이진 분류(good/bad)하는 경량 CNN 도구입니다.
멜 스펙트로그램 기반 3층 CNN (~83K params)으로, 독립 실행 또는 ASR 파이프라인 연동이 가능합니다.

```bash
# 1. 오디오 수집
uv run python tools/voice-checker/main.py import /path/to/audio

# 2. 라벨링 UI에서 good/bad 분류
uv run python tools/voice-checker/main.py serve

# 3. CNN 학습
uv run python tools/voice-checker/main.py train

# 4. 품질 예측
uv run python tools/voice-checker/main.py predict /path/to/audio -m tools/voice-checker/models/best_model.pth
```

**ASR 파이프라인 연동:**

```bash
uv run python scripts/data_preparation/asr_whisper.py \
  --voice-dir data/voice/dahwi \
  --voice-checker-model tools/voice-checker/models/best_model.pth
```

> Voice Checker는 루트 pyproject.toml의 의존성을 공유합니다. 별도 설치가 필요 없습니다.
```

CNN 통과 + ASR 텍스트 존재 시 라벨 상태를 `approved`로 자동 마킹합니다.

---

## 지원 모델 버전

| 버전 | 아키텍처 | 학습 스크립트 | 특징 |
|------|---------|-------------|------|
| v1, v2 | SynthesizerTrn (VITS) | `s2_train_vits.py` | v2에서 한국어 추가 |
| **v2Pro**, v2ProPlus | SynthesizerTrn + 화자 임베딩 | `s2_train_vits.py` | v2 비용으로 v3급 유사도 |
| v3, v4 | SynthesizerTrnV3 (CFM/DiT) | `s2_train_cfm.py` | 참조 오디오 충실, 감정 표현 우수 |

---

## 설정

`conf.example.yaml`을 복사하여 `conf.yaml`로 사용합니다.
기본값은 코드에 내장되어 있으며, 변경이 필요한 항목만 기재하면 됩니다.

```yaml
voices_dir: data/voice        # voice 디렉토리 루트
default_voice: dahwi           # 서버 시작 시 기본 로드할 voice (선택)

# service:                     # 서버 설정 (선택, CLI 인자로 오버라이드 가능)
#   host: 0.0.0.0
#   port: 9880

# voice_checker:               # Voice Checker 설정 (선택)
#   training:
#     epochs: 50
#     batch_size: 16
#   inference:
#     model_path: "tools/voice-checker/models/best_model.pth"
#   service:
#     port: 9890
```

---

## 프로젝트 구조

```
GPT-SoVITS-ko/
├── main.py                    # 진입점 (serve / pipeline)
├── conf.example.yaml          # 설정 템플릿
├── _setup_paths.py            # sys.path 단일 소스 관리
├── pyproject.toml             # 의존성 정의
├── src/
│   ├── config/                # Config, VoiceProfile
│   └── server/
│       ├── app.py             # FastAPI 팩토리
│       ├── context.py         # ServiceContext (DI 컨테이너)
│       └── routers/           # 도메인별 라우터 5개
├── scripts/
│   ├── data_preparation/      # Step 1: denoise, slice, UVR5, ASR
│   ├── preprocessing/         # Step 2: text, hubert, sv, semantic
│   ├── training/              # Step 3: GPT AR + SoVITS
│   ├── export/                # TorchScript, ONNX export
│   └── inference/             # Step 4: CLI 추론, 스트리밍
├── tools/
│   ├── voice-checker/         # CNN 오디오 품질 분류 도구
│   ├── label-review.html      # ASR 라벨 검수 UI
│   └── training/              # 학습 인프라 유틸
├── GPT_SoVITS/                # 모델 아키텍처 (원본 유지)
└── data/
    ├── voice/{character}/     # 캐릭터별 음성 데이터
    └── models/                # 프리트레인드 모델
```

---

## 라이선스

이 프로젝트는 **MIT License**로 배포됩니다.

원본 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)(MIT License, Copyright (c) 2024 RVC-Boss) 코드를 포함하고 있습니다.

### 자체 코드 (MIT)

`src/`, `scripts/`, `main.py`와 아래 `tools/` 내 파일:

- `tools/voice-checker/` — 오디오 품질 분류 도구
- `tools/training/` — 학습 인프라
- `tools/dl_utils.py` — 모델 다운로드 유틸
- `tools/label-review.html` — ASR 라벨 검수 UI

### 원본 GPT-SoVITS 유래 코드 (MIT, RVC-Boss)

| 컴포넌트 | 경로 |
|----------|------|
| 모델 아키텍처 | `GPT_SoVITS/` |
| 오디오 슬라이서 | `tools/slicer2.py` |
| 오디오 유틸 | `tools/my_utils.py`, `tools/audio_sr.py` |
| ASR 설정 | `tools/asr/` |
| 보컬 분리 (UVR5) | `tools/uvr5/` |
| BigVGAN (NVIDIA) | `GPT_SoVITS/BigVGAN/` (MIT + 하위 라이선스 — `incl_licenses/` 참조) |

### 외부 프로젝트

| 컴포넌트 | 라이선스 | 경로 |
|----------|---------|------|
| AP-BWE (Ye-Xin Lu) | MIT | `tools/AP_BWE_main/LICENSE` |
