#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.

# Note: The pretrained model path assumes the checkpoint is located at ../models/630k-audioset-best.pt
# You may need to adjust the --pretrained path if your model is located elsewhere.
# You can download the model from: https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt

python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "../models/630k-audioset-best.pt" \
    --dataset us8k \
    --dataset_root "../data/UrbanSound8K" \
    --task linear_probe \
    --train_split train \
    --test_split test \
    --output result_us8k_linear_probe.json \
    --batch_size 64 \
    --num_workers 4
