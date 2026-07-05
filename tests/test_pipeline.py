from __future__ import annotations

import torch

from multi_emo.data import MockIemocapDataset, collate_batch, parse_alignment_words
from multi_emo.metrics import classification_metrics
from multi_emo.preprocessing import AudioFeatureConfig, pad_or_trim
from multi_emo.splits import make_split
from multi_emo.train import train


def test_mock_collate_public_keys() -> None:
    dataset = MockIemocapDataset(3)
    batch = collate_batch([dataset[0], dataset[1]])
    assert set(["waveform", "mfcc", "spectrogram", "input_ids", "attention_mask", "label", "session", "speaker", "utterance_id"]).issubset(batch)
    assert batch["mfcc"].shape[-1] == 40
    assert batch["spectrogram"].shape[2] == 200


def test_speaker_independent_split_has_no_test_session_leakage() -> None:
    rows = MockIemocapDataset(20).rows
    train_idx, test_idx = make_split(rows, "speaker_independent", fold=0)
    train_sessions = {rows[idx]["session"] for idx in train_idx}
    test_sessions = {rows[idx]["session"] for idx in test_idx}
    assert train_sessions.isdisjoint(test_sessions)


def test_speaker_dependent_split_is_reproducible() -> None:
    rows = MockIemocapDataset(20).rows
    assert make_split(rows, "speaker_dependent", seed=123) == make_split(rows, "speaker_dependent", seed=123)


def test_preprocessing_padding() -> None:
    cfg = AudioFeatureConfig()
    waveform = pad_or_trim(torch.ones(100), cfg.segment_samples)
    assert waveform.shape == (48000,)
    assert waveform[:100].sum().item() == 100


def test_alignment_parser_gridtext(tmp_path) -> None:
    path = tmp_path / "utt.txt"
    path.write_text("0.00 0.50 hello\n0.50 1.00 world\n", encoding="utf-8")
    assert parse_alignment_words(path) == [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]


def test_metrics() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])
    metrics = classification_metrics(logits, targets)
    assert metrics["wa"] == 1.0
    assert metrics["ua"] == 0.5


def test_train_dry_run() -> None:
    cfg = {
        "data": {"mock_num_samples": 10},
        "model": {"use_pretrained_backbones": False},
        "train": {"seed": 1, "lr": 1e-4, "epochs": 1, "early_stopping_patience": 1},
        "eval": {"fold_seed": 42},
    }
    metrics = train(cfg, split="speaker_dependent", fold=0, dry_run=True)
    assert "wa" in metrics
    assert "ua" in metrics
