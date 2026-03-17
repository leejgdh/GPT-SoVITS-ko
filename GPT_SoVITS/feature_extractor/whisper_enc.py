import torch


def get_model():
    import whisper

    model = whisper.load_model("small", device="cpu")

    return model.encoder


def get_content(model=None, wav_16k_tensor=None):
    from whisper import log_mel_spectrogram, pad_or_trim

    dev = next(model.parameters()).device
    mel = log_mel_spectrogram(wav_16k_tensor).to(dev)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert mel.shape[-1] < 3000, "입력 오디오가 너무 깁니다. 30초 이내의 오디오만 허용됩니다."
    with torch.no_grad():
        feature = model(pad_or_trim(mel, 3000).unsqueeze(0))[:1, :feature_len, :].transpose(1, 2)
    return feature
