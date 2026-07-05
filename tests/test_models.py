from __future__ import annotations

import torch

from multi_emo.models import CMCA, MAFF, MultiEmoModel


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "waveform": torch.randn(batch_size, 48000),
        "mfcc": torch.randn(batch_size, 120, 40),
        "spectrogram": torch.randn(batch_size, 1, 200, 120),
        "input_ids": torch.randint(1, 100, (batch_size, 16)),
        "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        "label": torch.randint(0, 4, (batch_size,)),
    }


def test_maff_shape() -> None:
    module = MAFF(hidden_dim=128)
    out = module(torch.randn(2, 128), torch.randn(2, 128), torch.randn(2, 40, 768))
    assert out.shape == (2, 128)


def test_cmca_shape() -> None:
    module = CMCA(hidden_dim=128, seq_len=20, heads=2, self_attention_layers=2)
    out = module(torch.randn(2, 40, 768), torch.randn(2, 768))
    assert out.shape == (2, 128)


def test_model_forward_shapes() -> None:
    model = MultiEmoModel({"use_pretrained_backbones": False})
    out = model(make_batch(), return_features=True)
    assert out["logits"].shape == (2, 4)
    features = out["features"]
    assert features["mfcc"].shape == (2, 128)
    assert features["spec"].shape == (2, 128)
    assert features["maff"].shape == (2, 128)
    assert features["text"].shape == (2, 128)
    assert features["cmca"].shape == (2, 128)
