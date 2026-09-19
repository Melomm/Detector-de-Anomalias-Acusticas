"""DSP idêntico ao firmware: log-mel com posição temporal preservada."""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000
SAMPLES = 32000
FRAME = 400
HOP = 160
FFT = 512
MELS = 24
TIME_BINS = 24
FEATURES = MELS * TIME_BINS
STRIDE = 4000  # 250 ms entre decisões; janela de 2 s.
FRAMES = 1 + (SAMPLES - FRAME) // HOP


def mel_filters():
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    edges = 700.0 * (10 ** (np.linspace(mel(80), mel(7600), MELS + 2) / 2595.0) - 1)
    freq = np.arange(FFT // 2 + 1) * RATE / FFT
    filters = []
    for a, b, c in zip(edges, edges[1:], edges[2:]):
        filters.append(np.maximum(0, np.minimum((freq-a)/(b-a), (c-freq)/(c-b))))
    return np.asarray(filters, dtype=np.float32)


MEL_FILTERS = mel_filters()
HANN = np.hanning(FRAME).astype(np.float32)


def read_audio(path: Path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError(f"Áudio vazio ou inválido: {path}")
    if rate != RATE:
        divisor = gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor).astype(np.float32)
    return np.clip(audio, -1, 1)


def training_clip(path):
    audio = read_audio(path)
    # Não cortar silenciosamente a palavra final de um exemplo positivo.
    if len(audio) > SAMPLES + 160:
        raise ValueError(f"{path}: use clipes de até 2 s; recorte manualmente preservando o alvo.")
    audio = audio[:SAMPLES]
    left = (SAMPLES - len(audio)) // 2
    return np.pad(audio, (left, SAMPLES - len(audio) - left))


def extract(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape != (SAMPLES,):
        raise ValueError(f"Esperadas {SAMPLES} amostras mono, recebido {audio.shape}")
    # Remoção DC por frame, sem normalização individual de amplitude.
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
    centered = (frames - frames.mean(axis=1, keepdims=True)) * HANN
    power = np.abs(np.fft.rfft(centered, n=FFT)) ** 2 / FFT
    logmel = np.log(np.maximum(power @ MEL_FILTERS.T, 1e-10))
    pooled = np.stack([logmel[i * FRAMES // TIME_BINS:(i+1) * FRAMES // TIME_BINS].mean(axis=0)
                       for i in range(TIME_BINS)])
    rms = float(np.sqrt(np.mean(audio ** 2)))
    frequencies = np.arange(FFT // 2 + 1) * RATE / FFT
    centroid = float(np.mean((power @ frequencies) / np.maximum(power.sum(axis=1), 1e-10)))
    return pooled.astype(np.float32).ravel(), rms, centroid


def windows(audio):
    """Replay causal, incluindo cauda com padding (sem inventar janelas extras)."""
    if not len(audio):
        return
    for end in range(SAMPLES, max(SAMPLES, len(audio)) + STRIDE, STRIDE):
        start = end - SAMPLES
        clip = np.zeros(SAMPLES, dtype=np.float32)
        part = audio[start:min(end, len(audio))]
        clip[:len(part)] = part
        yield end / RATE, clip


def augment(audio, rng):
    result = audio.copy()
    # Só desloca para o espaço disponível; nunca corta o início/fim da fala.
    active = np.flatnonzero(np.abs(result) > 0.003)
    if len(active):
        lo = max(-int(active[0]), -3200)
        hi = min(SAMPLES - 1 - int(active[-1]), 3200)
        shift = int(rng.integers(lo, hi + 1))
        if shift > 0:
            result = np.pad(result[:-shift], (shift, 0))
        elif shift < 0:
            result = np.pad(result[-shift:], (0, -shift))
    result *= rng.uniform(0.65, 1.35)
    result += rng.normal(0, rng.uniform(0.00001, 0.001), SAMPLES).astype(np.float32)
    return np.clip(result, -1, 1)
