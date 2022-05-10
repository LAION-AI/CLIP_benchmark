#!/usr/bin/env python

"""Tests for `clip_benchmark` package."""

from clip_benchmark.cli import run

class base_args:
    dataset="dummy"
    split="test"
    model="ViT-B-32-quickgelu"
    pretrained="laion400m_e32"
    task="zeroshot_classification"
    amp=False
    num_workers=4
    batch_size=64
    dataset_root="root"
    output="result.json"
    verbose=True
    root="root"
    annotation_file=""

def test_base():
    run(base_args)

