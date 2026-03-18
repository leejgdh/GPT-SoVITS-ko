# GPT-SoVITS-ko

한국어 특화 음성 복제 TTS 서비스.
[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)(MIT) 를 기반으로, WebUI를 제거하고 **CLI 파이프라인 + REST API** 구조로 재설계했습니다.

> 소량의 음성 데이터(1~5분)로 화자의 음성을 학습하고, REST API로 실시간 합성할 수 있습니다.

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
uv sync
cp conf.example.yaml conf.yaml
```

---

## 사용법: 음성 학습부터 TTS 합성까지

### 1. 음성 데이터 준비

학습할 화자의 음성 파일(WAV, 16bit, 44.1kHz+)을 `data/voice/{name}/raw_audio/`에 넣습니다.

```bash
mkdir -p data/voice/dahwi/raw_audio
cp /path/to/recordings/*.wav data/voice/dahwi/raw_audio/
```

### 2. 파이프라인 실행

원커맨드로 데이터 준비 → 전처리 → 학습 → 추론을 일괄 실행합니다.

```bash
uv run python main.py pipeline \
  --voice-dir data/voice/dahwi \
  --output-text "합성 테스트용 텍스트" \
  --version v2Pro
```

파이프라인 내부 흐름:

```
Step 1: 데이터 준비  — 노이즈 제거 → 슬라이싱 → 보컬 분리 → ASR → 품질 분류
Step 2: 전처리      — 음소 추출 → HuBERT → 화자 임베딩 → Semantic 토큰
Step 3: 학습        — GPT AR + SoVITS Vocoder
Step 4: 추론        — 테스트 합성 + voice.yaml 자동 생성
```

각 step은 개별 실행할 수 있습니다. `uv run python main.py --help`로 전체 커맨드를 확인하세요.

### 3. ASR 라벨 검수 (선택)

Step 1 완료 후, Whisper가 생성한 자동 라벨을 웹 UI에서 검수할 수 있습니다.
step/pipeline 실행 시 서버가 백그라운드에서 자동으로 시작됩니다.

```
http://localhost:9880/review
```

- 오디오 재생 + 파형 시각화
- 라벨 상태 관리 (pending → approved / rejected)
- 텍스트 인라인 편집, 감정 매핑

### 4. 서버 실행

```bash
uv run python main.py              # 기본 포트 9880
uv run python main.py serve --port 8080
```

### 5. TTS 합성 요청

```bash
curl -X POST http://localhost:9880/tts \
  -H "Content-Type: application/json" \
  -d '{"voice": "dahwi", "text": "안녕하세요", "text_lang": "ko", "emotion": "default"}' \
  --output output.wav
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
| `speed_factor` | float | - | 속도 (기본: 1.0) |
| `temperature` | float | - | 샘플링 온도 (기본: 1.0) |
| `top_k` | int | - | Top-K 샘플링 (기본: 15) |

### 그 외 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /voices` | Voice 목록 조회 |
| `GET /voices/{name}` | Voice 상세 정보 |
| `GET /voices/{name}/labels` | ASR 라벨 목록 + 통계 |
| `GET /voices/{name}/emotions` | 감정 매핑 목록 |
| `GET /health` | 헬스 체크 |
| `GET /review` | ASR 라벨 검수 UI |

---

## Voice Profile

각 캐릭터의 음성 설정은 `data/voice/{name}/voice.yaml`로 관리합니다.
파이프라인 Step 4 완료 시 자동 생성되며, 감정 매핑은 API 또는 검수 UI를 통해 추가합니다.

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
```

---

## 지원 모델 버전

| 버전 | 아키텍처 | 특징 |
|------|---------|------|
| v2 | SynthesizerTrn (VITS) | 한국어 지원 |
| **v2Pro** | SynthesizerTrn + 화자 임베딩 | v2 비용으로 v3급 유사도 |
| v3, v4 | SynthesizerTrnV3 (CFM/DiT) | 감정 표현 우수 |

---

## 라이선스

MIT License.

[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT, Copyright (c) 2024 RVC-Boss) 를 기반으로 하며, 원본 코드는 `GPT_SoVITS/`, `tools/audio/`, `tools/asr/`, `tools/uvr5/`에 포함되어 있습니다. BigVGAN(`GPT_SoVITS/BigVGAN/`)은 NVIDIA MIT + 하위 라이선스를 따릅니다(`incl_licenses/` 참조). AP-BWE(`tools/AP_BWE_main/`)는 MIT License입니다.
