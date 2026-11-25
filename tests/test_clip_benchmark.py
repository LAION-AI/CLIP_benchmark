#!/usr/bin/env python

"""Tests for `clip_benchmark` package."""

import os
from clip_benchmark.cli import run
import logging
import pytest
import torch

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

def test_base():
    if torch.cuda.is_available():
        run(linear_probe_args)
    else:
        logging.warning("GPU acceleration is required for linear evaluation to ensure optimal performance and efficiency.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run(base_args)


def _parse_eval(argv, monkeypatch):
    import sys
    from clip_benchmark.cli import get_parser_args
    monkeypatch.setattr(sys, "argv", ["clip_benchmark", "eval"] + argv)
    _, args = get_parser_args()
    return args


@pytest.mark.parametrize("argv", [
    [],
    ["--model", "ViT-B-32-quickgelu", "--pretrained", "laion400m_e32"],
    ["--dataset", "cifar10"],
    ["--dataset", "cifar10", "--model", "ViT-B-32-quickgelu"],
    ["--dataset", "cifar10", "--pretrained", "laion400m_e32"],
    ["--dataset", "cifar10", "--model", "hf-hub:org/model", "ViT-B-32-quickgelu"],
    ["--dataset", "cifar10", "--model_type", "ja_clip"],
])
def test_eval_missing_required_args(argv, monkeypatch):
    with pytest.raises(SystemExit):
        _parse_eval(argv, monkeypatch)


@pytest.mark.parametrize("argv,model,pretrained", [
    (["--dataset", "cifar10", "--model", "ViT-B-32-quickgelu", "--pretrained", "laion400m_e32"], ["ViT-B-32-quickgelu"], ["laion400m_e32"]),
    (["--dataset", "cifar10", "--pretrained_model", "ViT-B-32-quickgelu,laion400m_e32"], None, None),
    (["--dataset", "cifar10", "--model", "hf-hub:org/model"], ["hf-hub:org/model"], [""]),
    (["--dataset", "cifar10", "--model_type", "ja_clip", "--pretrained", "rinna/japanese-cloob-vit-b-16"], [""], ["rinna/japanese-cloob-vit-b-16"]),
])
def test_eval_valid_args(argv, model, pretrained, monkeypatch):
    args = _parse_eval(argv, monkeypatch)
    assert args.model == model
    assert args.pretrained == pretrained
