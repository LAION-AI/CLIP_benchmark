# Audio Benchmarks for CLIP Benchmark

This document describes the audio modality support in CLIP Benchmark, enabling evaluation of audio-language models like CLAP.

## Supported Audio Datasets

| Dataset          | Description                            | Classes | Samples | Available Splits                             |
| ---------------- | -------------------------------------- | ------- | ------- | -------------------------------------------- |
| **ESC-50**       | Environmental sound classification     | 50      | 2,000   | `all`                                        |
| **UrbanSound8K** | Urban sound classification             | 10      | 8,732   | `train` (folds 1-9), `test` (fold 10), `all` |
| **GTZAN**        | Music genre recognition                | 10      | 1,000   | `all`                                        |
| **FSD50K**       | Freesound audio tagging (multi-label)  | 200     | 51,197  | `train` (dev set), `eval`                    |
| **VGGSounder**   | Audio events from videos (multi-label) | 309     | ~200k   | `all`                                        |

## Supported Evaluation Tasks

### Zero-Shot Classification

Evaluates how well the model classifies audio using only text descriptions of classes.

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "path/to/630k-audioset-best.pt" \
    --dataset esc50 \
    --dataset_root ./data/esc50 \
    --task zeroshot_classification \
    --output result_esc50.json \
    --split train
```

### Linear Probe

Trains a linear classifier on frozen audio embeddings.

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "path/to/630k-audioset-best.pt" \
    --dataset us8k \
    --dataset_root ./data/UrbanSound8K \
    --task linear_probe \
    --train_split train \
    --test_split test \
    --output result_us8k_lp.json \
    --batch_size 64
```

## Dataset-Specific Notes

### ESC-50

- Source: [HuggingFace `ashraq/esc50`](https://huggingface.co/datasets/ashraq/esc50)
- Automatically downloaded when first used
- Single-label classification

### UrbanSound8K

- Download from: https://urbansounddataset.weebly.com/urbansound8k.html
- Requires manual download and extraction
- Uses fold 10 as test set, and the rest as train set (normally 10 fold cross validation)

### GTZAN

- Download via Kaggle: `kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification`
- Automatically downloads if Kaggle CLI is installed

### FSD50K

- Download from: https://zenodo.org/records/4060432
- Multi-label classification (uses mAP metric)

### VGGSounder

- Requires `vggsounder` Python package
- Manuel download of vggsound necessary

## CLAP Model Checkpoints

Download pre-trained CLAP checkpoints from [HuggingFace](https://huggingface.co/lukewys/laion_clap):

| Checkpoint                             | Audio Encoder | Description              |
| -------------------------------------- | ------------- | ------------------------ |
| `630k-audioset-best.pt`                | HTSAT-tiny    | Trained on AudioSet 630k |
| `630k-audioset-fusion-best.pt`         | HTSAT-tiny    | With audio-text fusion   |
| `music_audioset_epoch_15_esc_90.14.pt` | HTSAT-tiny    | Fine-tuned on music      |

## Audio Preprocessing

All audio is:

- Resampled to **48kHz** (CLAP standard)
- Converted to **mono**
- Padded/cropped to dataset-specific target lengths

| Dataset    | Target Length   | Duration |
| ---------- | --------------- | -------- |
| ESC-50     | 240,000 samples | 5.0s     |
| US8K       | 192,000 samples | 4.0s     |
| GTZAN      | 144,000 samples | 3.0s     |
| FSD50K     | 192,000 samples | 4.0s     |
| VGGSounder | 384,000 samples | 8.0s     |
