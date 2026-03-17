# Voice Checker

오디오 품질 이진 분류(good/bad) CNN 모델. TTS Service의 ASR 라벨 자동 승인에 사용된다.

## 핵심 기술

| 영역 | 기술 |
|------|------|
| 모델 | AudioQualityCNN (Mel Spectrogram 기반) |
| 학습 | PyTorch, torchaudio |
| 증강 | Time/Frequency Masking, Noise Injection |
| 서버 | FastAPI, uvicorn (라벨링 UI) |

## 실행

```bash
# 오디오 파일 수집
python main.py import /path/to/audio/files

# 라벨링 UI 서버
python main.py serve
python main.py serve -v --host 0.0.0.0 --port 8200

# CNN 모델 학습
python main.py train

# 품질 예측
python main.py predict /path/to/audio.wav
python main.py predict /path/to/audio/dir -t 0.7    # 임계값 지정
```

## 워크플로

```
오디오 수집 (import) → 라벨링 UI (serve) → 모델 학습 (train) → TTS Service 연동 (predict)
```

## 설정

`conf.yaml` 참조. 주요 섹션: `audio`, `training`, `augmentation`, `inference`
