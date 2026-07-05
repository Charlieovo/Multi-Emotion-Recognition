from __future__ import annotations

import torch


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, pred in zip(targets.view(-1), preds.view(-1), strict=False):
        matrix[int(target), int(pred)] += 1
    return matrix


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> dict[str, object]:
    preds = logits.argmax(dim=-1)
    cm = confusion_matrix(preds.cpu(), targets.cpu(), num_classes)
    total = cm.sum().clamp_min(1)
    wa = cm.diag().sum().float() / total.float()
    per_class_total = cm.sum(dim=1).clamp_min(1)
    recall = cm.diag().float() / per_class_total.float()
    ua = recall.mean()

    f1_scores: list[float] = []
    for cls in range(num_classes):
        tp = cm[cls, cls].float()
        fp = cm[:, cls].sum().float() - tp
        fn = cm[cls, :].sum().float() - tp
        denom = (2 * tp + fp + fn).clamp_min(1)
        f1_scores.append(float((2 * tp / denom).item()))

    return {
        "wa": float(wa.item()),
        "ua": float(ua.item()),
        "per_class_f1": f1_scores,
        "confusion_matrix": cm.tolist(),
    }
