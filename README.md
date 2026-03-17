# GPT-SoVITS-ko

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)(MIT License, Copyright © 2024 RVC-Boss) 기반 한국어 특화 TTS 서비스.
WebUI 의존성을 제거하고 CLI 파이프라인 + REST API 구조로 재설계.

## 원본 대비 변경 사항

- Gradio WebUI 전체 제거 → FastAPI + ServiceContext 의존성 주입
- Voice Profile(`voice.yaml`) 기반 캐릭터 자체완결 구조
- 감정 프리셋(`EmotionRef`) — 감정별 참조 오디오 매핑, API에서 `emotion` 파라미터로 선택
- ASR 라벨 검수 시스템 — CRUD + 상태관리(pending/approved/rejected) + 웹 UI
- 4단계 CLI 파이프라인 — `python main.py pipeline` 원커맨드 실행
- 라우터 5분할 (tts, labels, emotions, voices, system)
- `_setup_paths.py`로 sys.path 단일 소스 관리
- `GPT_SoVITS/` 모델 아키텍처는 프리트레인 가중치 호환성을 위해 유지

## 핵심 기술

| 영역 | 기술 |
|------|------|
| TTS 모델 | GPT (AR Text-to-Semantic) + SoVITS (VITS Vocoder) |
| 지원 버전 | v1, v2, v2Pro, v2ProPlus, v3, v4 |
| 전처리 | Faster Whisper (ASR), FRCRN (노이즈 제거), UVR5 (보컬 분리) |
| 서버 | FastAPI, uvicorn |
| 품질 검증 | Voice Checker CNN 연동 (ASR 라벨 자동 승인) |

## 실행

### API 서버

```bash
uv run python main.py serve
uv run python main.py serve -c conf.yaml
uv run python main.py serve --host 0.0.0.0 --port 8080
```

### 원커맨드 파이프라인 (데이터 준비 → 전처리 → 학습 → 추론)

```bash
uv run python main.py pipeline \
  --voice-dir data/voice/dahwi \
  --output-text "합성할 텍스트"
```

### 개별 스텝

```bash
# Step 1: 데이터 준비
uv run python scripts/data_preparation/denoise.py --voice-dir data/voice/dahwi
uv run python scripts/data_preparation/slice_audio.py --voice-dir data/voice/dahwi
uv run python scripts/data_preparation/uvr5_separate.py --voice-dir data/voice/dahwi
uv run python scripts/data_preparation/asr_whisper.py --voice-dir data/voice/dahwi

# Step 2: 전처리
uv run python scripts/preprocessing/1-get-text.py --voice-dir data/voice/dahwi
uv run python scripts/preprocessing/2-get-hubert-wav32k.py --voice-dir data/voice/dahwi
uv run python scripts/preprocessing/3-get-semantic.py --voice-dir data/voice/dahwi

# Step 3: 학습
uv run python scripts/training/s1_train.py --voice-dir data/voice/dahwi --version v2Pro
uv run python scripts/training/s2_train_vits.py --voice-dir data/voice/dahwi --version v2Pro

# Step 4: 추론
uv run python scripts/inference/inference_cli.py \
  --voice-dir data/voice/dahwi --version v2Pro \
  --ref-audio data/voice/dahwi/step1/03_vocal/sample.flac \
  --ref-text "참조 텍스트" --text "합성할 텍스트"
```

## 데이터 디렉토리 구조

`data/voice/{character}/` 아래에 파이프라인 단계별 출력이 저장됩니다.
`dahwi/`는 예시 캐릭터이며, `voice.yaml.example`을 참고하세요.

```
data/
├── voice/
│   └── dahwi/                              # 캐릭터 폴더 (이름 자유)
│       ├── raw_audio/                      # 원본 음성 파일 (WAV, 16bit, 44.1kHz+)
│       ├── step1/                          # Step 1: 데이터 준비
│       │   ├── 01_denoise/                 # FRCRN 노이즈 제거 출력
│       │   ├── 02_sliced/                  # 무음 기반 슬라이싱 출력
│       │   ├── 03_vocal/                   # UVR5 보컬 분리 출력
│       │   └── 04_asr/                     # Whisper ASR 라벨 (.list)
│       ├── step2/{version}/                # Step 2: 전처리 (음소, HuBERT, semantic)
│       ├── step3/{version}/                # Step 3: 학습
│       │   ├── 01_gpt_logs/                # GPT 학습 로그 + 체크포인트
│       │   ├── 02_gpt_weights/             # GPT half-precision 가중치
│       │   ├── 03_sovits_logs/             # SoVITS 학습 로그 + 체크포인트
│       │   └── 04_sovits_weights/          # SoVITS half-precision 가중치
│       ├── step4/{version}/                # Step 4: 추론 출력
│       └── voice.yaml                      # Voice Profile (파이프라인 완료 시 자동 생성)
└── models/                                 # 프리트레인드 모델 (FRCRN, UVR5 등)
```

## API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/tts` | POST | TTS 합성 (스트리밍/일괄) |
| `/voices` | GET | voice 프로필 목록 |
| `/voices/{name}` | GET | voice 상세 |
| `/voices/{name}/labels` | GET/POST/PATCH | ASR 라벨 CRUD |
| `/voices/{name}/emotions/{emotion}` | GET/PUT/DELETE | 감정 매핑 CRUD |
| `/health` | GET | 헬스 체크 |

## 도구

### ASR 라벨 검수 UI

`tools/label-review.html`을 서버의 `/review` 엔드포인트에서 제공합니다.
ASR(Whisper)이 생성한 라벨을 검수하고 감정 매핑까지 수행하는 웹 UI입니다.

- 라벨 상태 관리: pending → approved / rejected
- 오디오 재생 + 파형 시각화
- 텍스트 인라인 편집
- 감정 프리셋 매핑 (label → emotion ref audio)
- 빈 텍스트 라벨 일괄 삭제

```bash
# 서버 실행 후 브라우저에서 접속
uv run python main.py serve
# → http://localhost:9880/review
```

### Voice Checker

`tools/voice-checker/`는 오디오 품질을 이진 분류(good/bad)하는 경량 CNN 도구입니다.
독립 실행 가능하며, ASR 파이프라인에 연동하면 품질 불량 오디오를 자동 rejected 처리합니다.

| 서브커맨드 | 설명 |
|-----------|------|
| `import` | 오디오 파일을 `data/`에 수집하고 `labels.json`에 등록 |
| `serve` | 라벨링 UI 서버 실행 (good/bad 분류용) |
| `train` | `labels.json` 기반 CNN 모델 학습 |
| `predict` | 학습된 모델로 오디오 품질 예측 |

```bash
cd tools/voice-checker

# 1. 오디오 수집
uv run python main.py import /path/to/audio

# 2. 라벨링 UI에서 good/bad 분류
uv run python main.py serve

# 3. CNN 학습
uv run python main.py train

# 4. 추론 (단독)
uv run python main.py predict /path/to/audio -m models/best_model.pth
```

ASR 파이프라인 연동:

```bash
# --voice-checker-model 옵션으로 CNN 필터링 적용
uv run python scripts/data_preparation/asr_whisper.py \
  --voice-dir data/voice/dahwi \
  --voice-checker-model tools/voice-checker/models/best_model.pth
```

## 지원 모델 버전

| 버전 | SoVITS 아키텍처 | 학습 스크립트 | 비고 |
|------|----------------|-------------|------|
| v1, v2 | SynthesizerTrn (VITS) | s2_train_vits.py | v2에서 한국어 추가 |
| v2Pro, v2ProPlus | SynthesizerTrn + 화자 임베딩 | s2_train_vits.py | 개발자 추천, v2 비용으로 v3급 유사도 |
| v3, v4 | SynthesizerTrnV3 (CFM/DiT) | s2_train_cfm.py | 참조 오디오 충실, 감정 표현 우수 |

## 설정

`conf.yaml` 참조. 주요 섹션: `tts` (모델 버전/가중치), `service` (호스트/포트)

## 라이선스

이 프로젝트 자체는 MIT License로 배포됩니다.

단, 원본 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)(MIT License) 코드를 포함하고 있으며, 출처별 라이선스는 아래를 참조하세요.

### 자체 코드 (MIT)

`src/`, `scripts/`, `main.py`와 아래 `tools/` 내 파일은 이 프로젝트의 자체 코드입니다.

- `tools/dl_utils.py` — 모델 다운로드 유틸
- `tools/training/` — 학습 인프라
- `tools/voice-checker/` — 오디오 품질 분류 도구
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
| 기타 서드파티 | 각 디렉토리 참조 | `GPT_SoVITS/` 내부 |