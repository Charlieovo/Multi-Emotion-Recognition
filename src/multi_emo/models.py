from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MultiEmoModelConfig:
    bert_name: str = "bert-base-uncased"
    wav2vec_name: str = "facebook/wav2vec2-base"
    use_pretrained_backbones: bool = False
    freeze_pretrained: bool = True
    use_maff: bool = True
    use_cmca: bool = True
    use_mfcc: bool = True
    use_spec: bool = True
    use_wav2vec: bool = True
    use_text: bool = True
    hidden_dim: int = 128
    cmca_seq_len: int = 20
    attention_heads: int = 2
    self_attention_layers: int = 2
    dropout: float = 0.5
    num_classes: int = 4


class MFCCEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(mfcc)
        return self.proj(self.dropout(output.mean(dim=1)))


class SpectrogramEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, use_pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        self.use_alexnet = False
        if use_pretrained:
            try:
                from torchvision.models import AlexNet_Weights, alexnet

                weights = AlexNet_Weights.DEFAULT
                self.alexnet = alexnet(weights=weights)
                self.alexnet.classifier[-1] = nn.Linear(self.alexnet.classifier[-1].in_features, hidden_dim)
                self.use_alexnet = True
                if freeze:
                    for name, param in self.alexnet.named_parameters():
                        param.requires_grad = name.startswith("classifier.6")
            except Exception:
                self.use_alexnet = False

        if not self.use_alexnet:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(32, hidden_dim)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.use_alexnet:
            resized = torch.nn.functional.interpolate(spectrogram, size=(224, 224), mode="bilinear", align_corners=False)
            rgb = resized.repeat(1, 3, 1, 1)
            return self.alexnet(rgb)
        features = self.conv(spectrogram).flatten(1)
        return self.proj(features)


class MockWaveformEncoder(nn.Module):
    def __init__(self, output_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=320, padding=12),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.net(waveform.unsqueeze(1)).transpose(1, 2)


class WaveformEncoder(nn.Module):
    def __init__(self, model_name: str, use_pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        self.use_transformers = False
        if use_pretrained:
            try:
                from transformers import Wav2Vec2Model

                self.model = Wav2Vec2Model.from_pretrained(model_name)
                self.use_transformers = True
                if freeze:
                    self.model.requires_grad_(False)
            except Exception:
                self.use_transformers = False
        if not self.use_transformers:
            self.model = MockWaveformEncoder()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.use_transformers:
            with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
                return self.model(waveform).last_hidden_state
        return self.model(waveform)


class MockTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 30522, output_dim: int = 768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim, padding_idx=0)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        if attention_mask is None:
            return self.norm(embedded[:, 0])
        mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, use_pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        self.use_transformers = False
        if use_pretrained:
            try:
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(model_name)
                self.use_transformers = True
                if freeze:
                    self.model.requires_grad_(False)
            except Exception:
                self.use_transformers = False
        if not self.use_transformers:
            self.model = MockTextEncoder()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_transformers:
            with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.model(input_ids, attention_mask)


class MAFF(nn.Module):
    def __init__(self, hidden_dim: int = 128, wav_dim: int = 768) -> None:
        super().__init__()
        self.mfcc_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spec_proj = nn.Linear(hidden_dim, hidden_dim)
        self.wav_proj = nn.Linear(wav_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, mfcc_vec: torch.Tensor, spec_vec: torch.Tensor, wav_seq: torch.Tensor) -> torch.Tensor:
        time_freq = torch.cat([self.activation(self.mfcc_proj(mfcc_vec)), self.activation(self.spec_proj(spec_vec))], dim=-1)
        wav_vec = wav_seq.mean(dim=1)
        wav_gate = self.wav_proj(wav_vec)
        return self.out_proj(time_freq * wav_gate)


class CMCA(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        wav_dim: int = 768,
        text_dim: int = 768,
        seq_len: int = 20,
        heads: int = 2,
        self_attention_layers: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.speech_proj = nn.Linear(wav_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.speech_to_text = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.text_to_speech = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm_st = nn.LayerNorm(hidden_dim)
        self.norm_ts = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=self_attention_layers)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, wav_seq: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        speech = self._resize_sequence(self.speech_proj(wav_seq), self.seq_len)
        text = self.text_proj(text_vec).unsqueeze(1).expand(-1, self.seq_len, -1)
        st, _ = self.speech_to_text(query=speech, key=text, value=text, need_weights=False)
        ts, _ = self.text_to_speech(query=text, key=speech, value=speech, need_weights=False)
        st = self.norm_st(st + text)
        ts = self.norm_ts(ts + speech)
        fused = torch.cat([st, ts], dim=-1)
        attended = self.self_attention(fused)
        return self.out_proj(attended.mean(dim=1))

    @staticmethod
    def _resize_sequence(sequence: torch.Tensor, seq_len: int) -> torch.Tensor:
        if sequence.shape[1] == seq_len:
            return sequence
        sequence = sequence.transpose(1, 2)
        sequence = torch.nn.functional.interpolate(sequence, size=seq_len, mode="linear", align_corners=False)
        return sequence.transpose(1, 2)


class BranchClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MultiEmoModel(nn.Module):
    def __init__(self, config: MultiEmoModelConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = MultiEmoModelConfig(**{k: v for k, v in config.items() if k in MultiEmoModelConfig.__annotations__})
        self.config = config or MultiEmoModelConfig()
        cfg = self.config

        self.mfcc_encoder = MFCCEncoder(cfg.hidden_dim, cfg.dropout)
        self.spec_encoder = SpectrogramEncoder(cfg.hidden_dim, cfg.use_pretrained_backbones, cfg.freeze_pretrained)
        self.wave_encoder = WaveformEncoder(cfg.wav2vec_name, cfg.use_pretrained_backbones, cfg.freeze_pretrained)
        self.text_encoder = TextEncoder(cfg.bert_name, cfg.use_pretrained_backbones, cfg.freeze_pretrained)
        self.maff = MAFF(cfg.hidden_dim)
        self.cmca = CMCA(
            hidden_dim=cfg.hidden_dim,
            seq_len=cfg.cmca_seq_len,
            heads=cfg.attention_heads,
            self_attention_layers=cfg.self_attention_layers,
        )
        self.text_proj = nn.Linear(768, cfg.hidden_dim)
        self.wav_proj = nn.Linear(768, cfg.hidden_dim)
        self.branch_projections = nn.ModuleDict(
            {
                "mfcc": nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                "spec": nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                "maff": nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                "text": nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                "cmca": nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            }
        )
        self.classifier = BranchClassifier(cfg.hidden_dim, cfg.num_classes, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor], return_features: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        mfcc_vec = self.mfcc_encoder(batch["mfcc"])
        spec_vec = self.spec_encoder(batch["spectrogram"])
        wav_seq = self.wave_encoder(batch["waveform"])
        text_vec_raw = self.text_encoder(batch["input_ids"], batch.get("attention_mask"))
        text_vec = self.text_proj(text_vec_raw)

        maff_vec = self.maff(mfcc_vec, spec_vec, wav_seq) if self.config.use_maff else self.wav_proj(wav_seq.mean(dim=1))
        cmca_vec = self.cmca(wav_seq, text_vec_raw) if self.config.use_cmca else 0.5 * (self.wav_proj(wav_seq.mean(dim=1)) + text_vec)

        branch_values = {
            "mfcc": mfcc_vec if self.config.use_mfcc else torch.zeros_like(mfcc_vec),
            "spec": spec_vec if self.config.use_spec else torch.zeros_like(spec_vec),
            "maff": maff_vec if self.config.use_wav2vec else torch.zeros_like(maff_vec),
            "text": text_vec if self.config.use_text else torch.zeros_like(text_vec),
            "cmca": cmca_vec if self.config.use_cmca else torch.zeros_like(cmca_vec),
        }
        projected = [self.branch_projections[name](value) for name, value in branch_values.items()]
        logits = self.classifier(torch.cat(projected, dim=-1))
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"logits": logits}
        if return_features:
            output["features"] = branch_values
        return output
