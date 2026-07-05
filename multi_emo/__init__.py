"""Source-layout shim so `python -m multi_emo.train` works before installation."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "multi_emo"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

from multi_emo.models import MultiEmoModel

__all__ = ["MultiEmoModel"]
