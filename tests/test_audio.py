"""Tests for audio support: loading a model directory with a tiny random model."""
import json

import pytest
import torch

open_clip = pytest.importorskip("open_clip")
pytest.importorskip("open_clip.audio")

from clip_benchmark.models import load_clip


def test_local_dir_model(tmp_path):
    """The model can be a directory with an OpenCLIP config and weights (the layout of a Hugging Face repository)."""
    model = open_clip.create_model("naflexclap_test")
    (tmp_path / "open_clip_config.json").write_text(json.dumps({"model_cfg": open_clip.get_model_config("naflexclap_test")}))
    torch.save(model.state_dict(), tmp_path / "open_clip_pytorch_model.bin")
    loaded, transform, tokenizer = load_clip("clap2", f"local-dir:{tmp_path}", "", None, "cpu")
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    with torch.inference_mode():
        features = loaded.encode_audio(transform([torch.randn(48000), torch.randn(96000)], [48000, 48000]))
        assert features.shape == (2, loaded.encode_text(tokenizer(["a dog barks"])).shape[-1])
