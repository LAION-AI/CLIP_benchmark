#!/usr/bin/env python

"""Tests for `clip_benchmark` package."""

import os
from clip_benchmark.cli import run
import logging
import torch
import pytest

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
    seed=0
    skip_load=False
    language="en"
    model_cache_dir=None
    save_clf=None
    load_clfs=[]
    model_type="open_clip"
    wds_cache_dir=None
    which="eval"
    skip_existing=False
    custom_template_file=None
    custom_classname_file=None
    distributed=False
    dump_classnames=False
    dump_templates=False

class linear_probe_args:
    dataset="dummy"
    split="test"
    train_split="split"
    val_split="val"
    val_proportion=None
    model="ViT-L-14"
    pretrained="openai"
    task="linear_probe"
    amp=False
    num_workers=4
    batch_size=256
    normalize=True
    dataset_root="root"
    output="result.json"
    verbose=True
    root="root"
    annotation_file=""
    seed=0
    feature_root="./"
    fewshot_k=-1
    fewshot_epochs=1
    fewshot_lr=0.5
    skip_load=False
    language="en"
    model_cache_dir=None
    save_clf=None
    load_clfs=[]
    model_type="open_clip"
    wds_cache_dir=None
    which="eval"
    skip_existing=False
    custom_template_file=None
    custom_classname_file=None
    distributed=False

class linear_probe_args:
    dataset="dummy"
    split="test"
    train_split="split"
    val_split="val"
    val_proportion=None
    model="ViT-L-14"
    pretrained="openai"
    task="linear_probe"
    amp=False
    num_workers=4
    batch_size=256
    normalize=True
    dataset_root="root"
    output="result.json"
    verbose=True
    root="root"
    annotation_file=""
    seed=0
    feature_root="./"
    fewshot_k=-1
    fewshot_epochs=1
    fewshot_lr=0.5
    skip_load=False
    language="en"
    model_cache_dir=None
    save_clf=None
    load_clfs=[]
    model_type="open_clip"
    wds_cache_dir=None
    which="eval"
    skip_existing=False
    custom_template_file=None
    custom_classname_file=None
    distributed=False


def test_linear_probe():
    if torch.cuda.is_available():
        run(linear_probe_args)
    else:
        logging.warning("GPU acceleration is required for linear evaluation to ensure optimal performance and efficiency.")


@pytest.mark.parametrize(
    "full_model_name",
    [
        "openai_clip:ViT-B/32",
        "open_clip:ViT-B-32/laion2b_s34b_b79k",
        "hf_clip:patrickjohncyh/fashion-clip",
    ],
)
def test_base(full_model_name):
    model_type, model_name = full_model_name.split(":")
    model, pretrained = model_name.split("/")
    base_args.model_type = model_type
    base_args.model = model
    base_args.pretrained = pretrained
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run(base_args)
