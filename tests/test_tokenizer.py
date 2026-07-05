from __future__ import annotations

from multi_emo.data import TextTokenizer


def test_fallback_tokenizer_shape() -> None:
    tokenizer = TextTokenizer(use_pretrained=False, max_length=8)
    input_ids, attention_mask = tokenizer.encode("hello emotional world")
    assert input_ids.shape == (8,)
    assert attention_mask.shape == (8,)
    assert input_ids[0].item() == 101
    assert attention_mask.sum().item() == 5
