#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
python3 -m clip_benchmark.cli eval \
    --model_type clap \
    --model "HTSAT-tiny" \
    --pretrained "../models/630k-audioset-best.pt" \
    --dataset us8k \
    --dataset_root /tmp/us8k \
    --task zeroshot_classification \
    --output result_us8k.json \
    --split all
