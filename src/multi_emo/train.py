from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from multi_emo.config import load_config
from multi_emo.data import IemocapDataset, MockIemocapDataset, TextTokenizer, collate_batch, read_metadata_csv
from multi_emo.metrics import classification_metrics
from multi_emo.models import MultiEmoModel
from multi_emo.preprocessing import AudioFeatureConfig
from multi_emo.splits import make_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(cfg: dict[str, Any], *, split: str, fold: int, dry_run: bool) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    if dry_run or not data_cfg.get("metadata_csv"):
        dataset = MockIemocapDataset(int(data_cfg.get("mock_num_samples", 16)))
        train_idx, val_idx = make_split(dataset.rows, split, fold=fold, seed=int(cfg.get("eval", {}).get("fold_seed", 42)))
    else:
        rows = read_metadata_csv(data_cfg["metadata_csv"])
        train_idx, val_idx = make_split(rows, split, fold=fold, seed=int(cfg.get("eval", {}).get("fold_seed", 42)))
        feature_cfg = AudioFeatureConfig(
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
            segment_seconds=float(data_cfg.get("segment_seconds", 3.0)),
            min_seconds=float(data_cfg.get("min_seconds", 1.0)),
        )
        model_cfg = cfg.get("model", {})
        tokenizer = TextTokenizer(
            model_name=str(model_cfg.get("bert_name", "bert-base-uncased")),
            use_pretrained=bool(model_cfg.get("use_pretrained_backbones", True)),
        )
        dataset = IemocapDataset(
            rows,
            root=data_cfg.get("root", ""),
            alignment_dir=data_cfg.get("alignment_dir", ""),
            feature_cfg=feature_cfg,
            tokenizer=tokenizer,
        )

    batch_size = 2 if dry_run else int(train_cfg.get("batch_size", 64))
    num_workers = 0 if dry_run else int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(
    model: MultiEmoModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = torch.nn.CrossEntropyLoss()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        labels = batch["label"]
        with torch.set_grad_enabled(is_train):
            logits = model(batch)["logits"]
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        count += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / max(count, 1)
    return metrics


def train(cfg: dict[str, Any], *, split: str, fold: int, dry_run: bool = False) -> dict[str, object]:
    train_cfg = cfg.get("train", {})
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")
    model_cfg = dict(cfg.get("model", {}))
    if dry_run:
        model_cfg["use_pretrained_backbones"] = False
    model = MultiEmoModel(model_cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg, split=split, fold=fold, dry_run=dry_run)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = 1 if dry_run else int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    best_ua = -1.0
    stale = 0
    best_metrics: dict[str, object] = {}
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    if not dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for _epoch in range(epochs):
        max_batches = 2 if dry_run else None
        run_epoch(model, train_loader, optimizer=optimizer, device=device, max_batches=max_batches)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, max_batches=max_batches)
        if float(val_metrics["ua"]) > best_ua:
            best_ua = float(val_metrics["ua"])
            best_metrics = val_metrics
            stale = 0
            if not dry_run:
                torch.save({"model": model.state_dict(), "config": cfg, "metrics": val_metrics}, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Emo CMCA + MAFF.")
    parser.add_argument("--config", default="configs/iemocap_cmca_maff.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["speaker_independent", "speaker_dependent"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.split or cfg.get("eval", {}).get("split", "speaker_independent")
    metrics = train(cfg, split=split, fold=args.fold, dry_run=args.dry_run)
    print(metrics)


if __name__ == "__main__":
    main()
