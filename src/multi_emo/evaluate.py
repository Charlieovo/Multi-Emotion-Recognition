from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multi_emo.config import load_config
from multi_emo.models import MultiEmoModel
from multi_emo.train import build_dataloaders, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Multi-Emo CMCA + MAFF.")
    parser.add_argument("--config", default="configs/iemocap_cmca_maff.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["speaker_independent", "speaker_dependent"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.split or cfg.get("eval", {}).get("split", "speaker_independent")
    dry_run = args.dry_run or not cfg.get("data", {}).get("metadata_csv")
    model_cfg = dict(cfg.get("model", {}))
    if dry_run:
        model_cfg["use_pretrained_backbones"] = False
    model = MultiEmoModel(model_cfg)
    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    _, val_loader = build_dataloaders(cfg, split=split, fold=args.fold, dry_run=dry_run)
    metrics = run_epoch(model, val_loader, optimizer=None, device=torch.device("cpu"), max_batches=2 if dry_run else None)
    print(metrics)


if __name__ == "__main__":
    main()
