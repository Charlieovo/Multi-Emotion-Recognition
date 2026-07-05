#!/usr/bin/env python
"""Create the metadata CSV expected by the training pipeline.

This helper is intentionally conservative because IEMOCAP directory layouts vary.
If auto-discovery does not match your copy, write a CSV with:
utterance_id,audio_path,text,label,session,speaker
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="IEMOCAP root directory")
    parser.add_argument("--output", default="metadata_iemocap.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    wavs = sorted(root.rglob("*.wav"))
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "audio_path", "text", "label", "session", "speaker"])
        writer.writeheader()
        for wav in wavs:
            writer.writerow(
                {
                    "utterance_id": wav.stem,
                    "audio_path": str(wav.relative_to(root)),
                    "text": "",
                    "label": "",
                    "session": next((part for part in wav.parts if part.lower().startswith("ses")), ""),
                    "speaker": wav.stem.split("_")[0],
                }
            )
    print(f"Wrote skeleton metadata for {len(wavs)} wav files to {args.output}. Fill text and label columns before training.")


if __name__ == "__main__":
    main()
