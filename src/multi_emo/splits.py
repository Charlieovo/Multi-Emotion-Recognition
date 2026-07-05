from __future__ import annotations

import random
from collections.abc import Sequence


def speaker_independent_split(rows: Sequence[dict[str, object]], fold: int) -> tuple[list[int], list[int]]:
    sessions = sorted({str(row["session"]) for row in rows})
    if not sessions:
        raise ValueError("No sessions found for speaker-independent split.")
    test_session = sessions[fold % len(sessions)]
    train_idx: list[int] = []
    test_idx: list[int] = []
    for idx, row in enumerate(rows):
        if str(row["session"]) == test_session:
            test_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, test_idx


def speaker_dependent_split(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(len(indices) * test_fraction)))
    test_idx = sorted(indices[:test_size])
    train_idx = sorted(indices[test_size:])
    return train_idx, test_idx


def make_split(
    rows: Sequence[dict[str, object]],
    split: str,
    *,
    fold: int = 0,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if split == "speaker_independent":
        return speaker_independent_split(rows, fold)
    if split == "speaker_dependent":
        return speaker_dependent_split(rows, seed=seed + fold)
    raise ValueError(f"Unsupported split: {split}")
