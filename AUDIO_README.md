# Audio Benchmarks for CLIP Benchmark

This document describes the audio modality support in CLIP Benchmark, enabling evaluation of audio-language models like CLAP. Note that the current version only supports webdatasets for audio modality datasets.

## Supported Audio Datasets

| Dataset          | Description                           | Task                                      | Classes | Available Splits                      |
| ---------------- | ------------------------------------- | ----------------------------------------- | ------- | ------------------------------------- |
| **ESC-50**       | Environmental sound classification    | `zeroshot_classification`, `linear_probe` | 50      | `train`, `test`                       |
| **UrbanSound8K** | Urban sound classification            | `zeroshot_classification`, `linear_probe` | 10      | `train` (folds 1-9), `test` (fold 10) |
| **GTZAN**        | Music genre recognition               | `zeroshot_classification`                 | 10      | `test`                                |
| **FSD50K**       | Freesound audio tagging (multi-label) | `zeroshot_classification`, `linear_probe` | 200     | `train`, `validation`, `test`         |
| **AudioCaps**    | Audio captioning dataset              | `zeroshot_retrieval`                      | -       | `train`, `validation`, `test`         |
| **Clotho**       | Audio captioning dataset              | `zeroshot_retrieval`                      | -       | `train`, `validation`, `test`         |

## Benchmark Results

### Zero-Shot Classification

Results using CLAP (HTSAT-tiny) on audio datasets:

| Dataset                   | Fusion |  Acc@1 |  Acc@5 |    mAP |
| ------------------------- | :----: | -----: | -----: | -----: |
| ESC-50                    |        | 92.50% | 99.50% |      - |
| ESC-50                    |   ✓    | 92.75% | 98.75% |      - |
| ESC-50 (no overlap)       |        | 91.03% | 99.35% |      - |
| ESC-50 (no overlap)       |   ✓    | 89.65% | 98.79% |      - |
| UrbanSound8K              |        | 80.65% | 96.54% |      - |
| UrbanSound8K              |   ✓    | 76.94% | 97.13% |      - |
| UrbanSound8K (no overlap) |        | 75.60% | 97.02% |      - |
| UrbanSound8K (no overlap) |   ✓    | 75.95% | 98.40% |      - |
| GTZAN                     |        | 53.65% | 74.97% |      - |
| GTZAN                     |   ✓    | 34.53% | 67.87% |      - |
| FSD50K                    |        |      - |      - | 55.97% |
| FSD50K                    |   ✓    |      - |      - | 56.90% |
| VGGSound                  |        |      - |      - | 24.87% |
| VGGSound                  |   ✓    |      - |      - | 17.23% |
| VGGSounder                |        |      - |      - | 26.65% |
| VGGSounder                |   ✓    |      - |      - | 18.85% |

### Linear Probe

Results using CLAP (HTSAT-tiny) on audio datasets:

| Dataset      | Fusion |  Acc@1 |    mAP |    Gain (vs ZS) |
| ------------ | :----: | -----: | -----: | ------: |
| ESC-50       |        | 97.00% |      - |  +4.50% |
| ESC-50       |   ✓    | 95.75% |      - |  +3.00% |
| UrbanSound8K |        | 88.89% |      - |  +8.24% |
| UrbanSound8K |   ✓    | 88.29% |      - | +11.35% |
| FSD50K       |        |      - | 67.52% | +11.55% |
| FSD50K       |   ✓    |      - | 68.06% | +11.16% |
| VGGSound     |        |      - | 52.47% | +27.60% |
| VGGSound     |   ✓    |      - | 50.90% | +33.67% |

### Zero-Shot Retrieval

Results using CLAP (HTSAT-tiny) on audio datasets:

| Dataset   | Fusion | Audio R@5 |
| --------- | :----: | --------: |
| AudioCaps |        |    79.34% |
| AudioCaps |   ✓    |    76.64% |
| Clotho    |        |    42.11% |
| Clotho    |   ✓    |    43.16% |

> **Note**: Fusion models use `630k-audioset-fusion-best.pt`, standard models use `630k-audioset-best.pt`.

## Supported Evaluation Tasks

### Zero-Shot Classification

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "/path/to/630k-audioset-best.pt" \
    --dataset wds/fsd50k \
    --num_workers 1 \
    --dataset_root /path/to/fsd50k \
    --task zeroshot_classification \
    --output benchmark/result_wds_fsd50k.json \
    --split test \
    --modality audio \
    --no_amp
```

### Linear Probe

Trains a linear classifier on frozen audio embeddings.

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "/path/to/630k-audioset-best.pt" \
    --dataset wds/fsd50k \
    --dataset_root "/path/to/web-dataset_conv/fsd50k" \
    --task linear_probe \
    --train_split train \
    --test_split test \
    --output benchmark/result_fsd50k_linear_probe.json \
    --batch_size 64 \
    --num_workers 1 \
    --modality audio \
    --no_amp
```

### Zero-Shot Retrieval

Evaluates retrieval performance (e.g., Audio-to-Text or Text-to-Audio).

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "/path/to/630k-audioset-best.pt" \
    --dataset wds/audiocaps \
    --num_workers 1 \
    --dataset_root /path/to/web-dataset_conv/wds_audiocaps \
    --task zeroshot_retrieval \
    --output benchmark/result_wds_audiocaps_retrieval.json \
    --split test \
    --modality audio \
    --no_amp
```

### Build result CSV:

```bash
python3 -m clip_benchmark.cli build ./results/benchmark/*.json --output benchmark/benchmark.csv
```

## Run multi-eval:

```bash
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --pretrained_model path/to/models.txt" \
    --dataset path/to/datasets.txt \
    --dataset_root path/to/{dataset} \
    --task zeroshot_classification \
    --output benchmark/{dataset}_{pretrained}_{model}_{language}_{task}.json \
    --modality audio \
    --num_workers 1 \
    --no_amp
```

> **Note:** Replace the task and models/dataset file accordingly.

## Configuration Files

These configuration files are referenced by the multi-eval scripts.

### `models.txt`

```text
HTSAT-tiny,../models/630k-audioset-best.pt
HTSAT-tiny,../models/630k-audioset-fusion-best.pt
```

### `classification_datasets.txt`

```text
wds/esc50
wds/UrbanSound8K
wds/gtzan
wds/fsd50k
```

### `linear_probe_datasets.txt`

```text
wds/esc50
wds/UrbanSound8K
wds/fsd50k
```

### `retrieval_datasets.txt`

```text
wds/clotho
wds/audiocaps
```
