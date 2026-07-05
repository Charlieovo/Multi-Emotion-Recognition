from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from multi_emo.preprocessing import AudioFeatureConfig, make_mock_features, pad_or_trim

LABEL_TO_ID = {
    "neutral": 0,
    "neu": 0,
    "happy": 1,
    "happiness": 1,
    "excited": 1,
    "excitement": 1,
    "angry": 2,
    "anger": 2,
    "sad": 3,
    "sadness": 3,
}


def normalize_label(label: str) -> int:
    key = label.strip().lower()
    if key not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label {label!r}. Expected one of {sorted(LABEL_TO_ID)}")
    return LABEL_TO_ID[key]


def read_metadata_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"utterance_id", "audio_path", "text", "label", "session", "speaker"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")
    for row in rows:
        row["label_id"] = normalize_label(str(row["label"]))
    return rows


class MockIemocapDataset(Dataset):
    def __init__(self, num_samples: int = 16) -> None:
        self.rows = [
            {
                "utterance_id": f"mock_{idx:04d}",
                "session": f"Ses{idx % 5 + 1:02d}",
                "speaker": f"spk_{idx % 4}",
                "label_id": idx % 4,
                "text": "mock emotional utterance",
            }
            for idx in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        features = make_mock_features(seed=idx)
        return {
            **features,
            "label": torch.tensor(row["label_id"], dtype=torch.long),
            "session": row["session"],
            "speaker": row["speaker"],
            "utterance_id": row["utterance_id"],
        }


class TextTokenizer:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 64, use_pretrained: bool = True) -> None:
        self.max_length = max_length
        self.tokenizer = None
        if use_pretrained:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                self.tokenizer = None

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"].squeeze(0).long(), encoded["attention_mask"].squeeze(0).long()
        tokens = text.strip().split()[: self.max_length - 2]
        ids = [101]
        ids.extend((abs(hash(token)) % 30000) + 100 for token in tokens)
        ids.append(102)
        mask = [1] * len(ids)
        if len(ids) < self.max_length:
            pad = self.max_length - len(ids)
            ids.extend([0] * pad)
            mask.extend([0] * pad)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


class IemocapDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        root: str | Path = "",
        alignment_dir: str | Path = "",
        feature_cfg: AudioFeatureConfig | None = None,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.rows = [
            row
            for row in rows
            if not row.get("duration") or float(row["duration"]) >= (feature_cfg or AudioFeatureConfig()).min_seconds
        ]
        self.root = Path(root) if root else Path(".")
        self.alignment_dir = Path(alignment_dir) if alignment_dir else None
        self.feature_cfg = feature_cfg or AudioFeatureConfig()
        self.tokenizer = tokenizer or TextTokenizer(use_pretrained=False)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the audio extra to load real IEMOCAP audio.") from exc

        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        waveform_np, sample_rate = sf.read(audio_path)
        if sample_rate != self.feature_cfg.sample_rate:
            try:
                import librosa
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install librosa to resample audio.") from exc
            waveform_np = librosa.resample(waveform_np.astype("float32"), orig_sr=sample_rate, target_sr=self.feature_cfg.sample_rate)
        waveform = torch.as_tensor(waveform_np, dtype=torch.float32).flatten()
        if waveform.numel() < self.feature_cfg.min_samples:
            raise ValueError(f"Utterance {row['utterance_id']} is shorter than the configured minimum.")

        from multi_emo.preprocessing import compute_audio_features

        waveform = pad_or_trim(waveform, self.feature_cfg.segment_samples)
        mfcc, spectrogram = compute_audio_features(waveform.numpy(), self.feature_cfg)

        token_ids, attention_mask = self.tokenizer.encode(self._resolve_segment_text(row))
        return {
            "waveform": waveform,
            "mfcc": mfcc,
            "spectrogram": spectrogram,
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "session": row["session"],
            "speaker": row["speaker"],
            "utterance_id": row["utterance_id"],
        }

    def _resolve_segment_text(self, row: dict[str, Any]) -> str:
        if self.alignment_dir is None:
            return str(row.get("text", ""))
        utterance_id = str(row["utterance_id"])
        for suffix in [".TextGrid", ".textgrid", ".grid", ".txt"]:
            path = self.alignment_dir / f"{utterance_id}{suffix}"
            if path.exists():
                words = parse_alignment_words(path)
                segment_words = [
                    word
                    for start, end, word in words
                    if word and start < self.feature_cfg.segment_seconds and end > 0.0
                ]
                if segment_words:
                    return " ".join(segment_words)
        return str(row.get("text", ""))


def parse_alignment_words(path: str | Path) -> list[tuple[float, float, str]]:
    """Parse a minimal MFA TextGrid or whitespace GridText word alignment file."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    triples: list[tuple[float, float, str]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                triples.append((float(parts[0]), float(parts[1]), " ".join(parts[2:]).strip('"')))
            except ValueError:
                pass
    if triples:
        return triples

    current_start: float | None = None
    current_end: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("xmin ="):
            current_start = _parse_textgrid_float(stripped)
        elif stripped.startswith("xmax ="):
            current_end = _parse_textgrid_float(stripped)
        elif stripped.startswith("text =") and current_start is not None and current_end is not None:
            word = stripped.split("=", 1)[1].strip().strip('"')
            if word:
                triples.append((current_start, current_end, word))
            current_start = None
            current_end = None
    return triples


def _parse_textgrid_float(line: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    if match is None:
        raise ValueError(f"Cannot parse TextGrid float from {line!r}")
    return float(match.group(0))


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    tensor_keys = ["waveform", "mfcc", "spectrogram", "input_ids", "attention_mask", "label"]
    for key in tensor_keys:
        values = [sample[key] for sample in samples]
        if key in {"mfcc", "spectrogram", "input_ids", "attention_mask"}:
            batch[key] = _pad_stack(values)
        else:
            batch[key] = torch.stack(values)
    for key in ["session", "speaker", "utterance_id"]:
        batch[key] = [sample[key] for sample in samples]
    return batch


def _pad_stack(values: list[torch.Tensor]) -> torch.Tensor:
    max_shape = [max(value.shape[dim] for value in values) for dim in range(values[0].ndim)]
    padded = []
    for value in values:
        pad_spec: list[int] = []
        for dim in reversed(range(value.ndim)):
            pad_spec.extend([0, max_shape[dim] - value.shape[dim]])
        padded.append(torch.nn.functional.pad(value, pad_spec))
    return torch.stack(padded)
