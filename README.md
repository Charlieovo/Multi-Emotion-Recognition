# Multi-Emo CMCA + MAFF Reproduction

This repository is a PyTorch reproduction scaffold for the paper **"Multimodal emotion recognition using cross-modal convolutional attention and multi-acoustic feature fusion"**.

The first version focuses on a runnable, inspectable implementation of the paper pipeline:

- MFCC -> 2-layer BiLSTM -> 128-d utterance vector
- Spectrogram -> AlexNet-style encoder -> 128-d utterance vector
- Waveform -> Wav2Vec 2.0 -> frame sequence
- Text -> BERT CLS vector
- MAFF: MFCC/Spectrogram fusion gated by Wav2Vec via Hadamard product
- CMCA: convolutional speech/text alignment, bidirectional cross-attention, and shallow self-attention
- Classifier over five branches: MFCC, Spectrogram, MAFF, Text, CMCA

The expected paper-level targets on IEMOCAP are approximately:

- Speaker-independent LOSO: `75.70 WA / 77.28 UA`
- Speaker-dependent random split: `78.50 WA / 79.62 UA`

Actual numbers depend on IEMOCAP preprocessing, alignment quality, split exactness, and pretrained checkpoint versions.

## Install

```bash
python -m pip install -e ".[dev]"
```

For real audio preprocessing and pretrained backbones:

```bash
python -m pip install -e ".[audio,pretrained,dev]"
```

## Quick Smoke Test

No IEMOCAP data is required:

```bash
python -m multi_emo.train --config configs/iemocap_cmca_maff.yaml --dry-run
pytest
```

Dry-run uses a deterministic mock dataset and lightweight backbone substitutes so the full training/evaluation path can be checked without downloading HuggingFace or torchvision weights.

## Data Format

Create a metadata CSV with at least these columns:

```text
utterance_id,audio_path,text,label,session,speaker
```

Accepted labels are:

- `neutral`
- `happy`, `happiness`, `excited`, `excitement`
- `angry`, `anger`
- `sad`, `sadness`

Happy and Excited are merged into the `happy` class, matching the paper's four-class IEMOCAP setup:

```text
neutral: 1708
happy + excited: 1636
anger: 1103
sad: 1084
```

The paper uses Montreal Forced Aligner/GridText files to align text to 3-second audio segments. This scaffold accepts `data.alignment_dir`, but if no alignment is provided it falls back to the full utterance text. That fallback is useful for engineering and smoke tests, but it is not a strict reproduction of the paper's segment-level text alignment. When HuggingFace tokenizers are available, text is encoded with `model.bert_name`; otherwise a deterministic whitespace/hash tokenizer is used only as a fallback.

## Training

Speaker-independent leave-one-session-out:

```bash
python -m multi_emo.train \
  --config configs/iemocap_cmca_maff.yaml \
  --split speaker_independent \
  --fold 0
```

Speaker-dependent random split:

```bash
python -m multi_emo.train \
  --config configs/iemocap_cmca_maff.yaml \
  --split speaker_dependent \
  --fold 0
```

Evaluation from a checkpoint:

```bash
python -m multi_emo.evaluate \
  --config configs/iemocap_cmca_maff.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split speaker_independent \
  --fold 0
```

## Configuration

The public config fields intentionally mirror the implementation plan:

- `data.root`
- `data.metadata_csv`
- `data.alignment_dir`
- `model.bert_name`
- `model.wav2vec_name`
- `train.batch_size`
- `train.lr`
- `train.epochs`
- `eval.split`
- `eval.fold_seed`

Additional flags control ablations and engineering behavior:

- `model.use_maff`
- `model.use_cmca`
- `model.use_mfcc`
- `model.use_spec`
- `model.use_wav2vec`
- `model.use_text`
- `model.use_pretrained_backbones`

The pretrained backbones are frozen by default when enabled. Only BiLSTM, MAFF, CMCA, projection layers, and the classifier are trained, matching the paper's lightweight training setup.

## Preprocessing Details

When real audio dependencies are installed, preprocessing follows the paper settings:

- 16 kHz audio
- discard utterances shorter than 1 second
- 3-second segments with zero padding
- HTK-style 40-dimensional MFCC
- Hamming window
- 10 ms frame shift
- 800-sample DFT
- spectrogram first 200 frequency bins

## Repository Layout

```text
configs/                     Experiment configs
scripts/                     Utility scripts
src/multi_emo/               Package source
tests/                       Unit and smoke tests
```
