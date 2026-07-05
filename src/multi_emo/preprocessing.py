from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    min_seconds: float = 1.0
    n_mfcc: int = 40
    n_fft: int = 800
    hop_length: int = 160
    spec_bins: int = 200

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)

    @property
    def min_samples(self) -> int:
        return int(self.sample_rate * self.min_seconds)


def pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.ndim != 1:
        waveform = waveform.flatten()
    if waveform.numel() >= length:
        return waveform[:length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.numel()))


def compute_audio_features(waveform: np.ndarray, cfg: AudioFeatureConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute paper-style MFCC and spectrogram features.

    librosa is imported lazily so tests can run without optional audio deps.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError("Install the audio extra to compute real MFCC/spectrogram features.") from exc

    mfcc = librosa.feature.mfcc(
        y=waveform.astype(np.float32),
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hamming",
        htk=True,
    ).T
    spec = np.abs(
        librosa.stft(
            waveform.astype(np.float32),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window="hamming",
        )
    )[: cfg.spec_bins]
    return torch.from_numpy(mfcc).float(), torch.from_numpy(spec[None, :, :]).float()


def make_mock_features(
    *,
    waveform_length: int = 48000,
    mfcc_frames: int = 120,
    spec_width: int = 120,
    token_count: int = 16,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "waveform": torch.randn(waveform_length, generator=generator),
        "mfcc": torch.randn(mfcc_frames, 40, generator=generator),
        "spectrogram": torch.randn(1, 200, spec_width, generator=generator),
        "input_ids": torch.randint(1, 100, (token_count,), generator=generator),
        "attention_mask": torch.ones(token_count, dtype=torch.long),
    }
