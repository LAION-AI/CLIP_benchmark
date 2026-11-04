"""HEAR (Holistic Evaluation of Audio Representations): model interface for heareval, which embeds and scores a task."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torchvision.datasets.utils import download_and_extract_archive

PREFIX = "hear/"
# The 48 kHz tasks of the HEAR NeurIPS 2021 release (2021.3) on Zenodo, except Beehive and Maestro.
TASKS = [
    "beijing_opera-v1.0-hear2021-full", "dcase2016_task2-hear2021-full", "esc50-v2.0.0-full", "fsd50k-v1.0-full",
    "gunshot_triangulation-v1.0-full", "libricount-v1.0.0-hear2021-full", "mridangam_stroke-v1.5-full",
    "mridangam_tonic-v1.5-full", "nsynth_pitch-v2.2.3-50h", "nsynth_pitch-v2.2.3-5h", "speech_commands-v0.0.2-5h",
    "speech_commands-v0.0.2-full", "tfds_crema_d-1.0.0-full", "tfds_gtzan-1.0.0-full",
    "tfds_gtzan_music_speech-1.0.0-full", "vox_lingua_top10-hear2021-full", "vocal_imitation-v1.1.3-full",
]
ZENODO = "https://zenodo.org/records/6332517/files/hear2021-{}-48000.tar.gz"


def dataset_names():
    return [PREFIX + name for name in TASKS]


def download(name, root):
    """Fetch and unpack a task from Zenodo into `root/<name>`."""
    Path(root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=root) as temporary:
        download_and_extract_archive(ZENODO.format(name), temporary, remove_finished=True)
        # the archives unpack to tasks/<name>, some of them inside a hear-2021.0.6/ directory
        source, = [*Path(temporary).glob(f"tasks/{name}"), *Path(temporary).glob(f"*/tasks/{name}")]
        source.rename(Path(root) / name)


class HEARModel(torch.nn.Module):
    def __init__(self, model, transform):
        super().__init__()
        self.model, self.transform = model, transform
        self.sample_rate = 48000
        self.scene_embedding_size = get_scene_embeddings(torch.zeros(1, self.sample_rate), self).shape[-1]
        self.timestamp_embedding_size = self.scene_embedding_size


def load_model(model, transform):
    return HEARModel(model, transform).eval()


@torch.inference_mode()
def get_scene_embeddings(audio, model):
    return model.model.encode_audio(model.transform(audio, [model.sample_rate] * len(audio)))


def get_timestamp_embeddings(audio, model):
    # One embedding of a centered 1 s window every 50 ms.
    window = model.sample_rate
    audio = torch.nn.functional.pad(audio, (window // 2, window - window // 2))
    starts = range(0, audio.shape[-1] - window + 1, window // 20)
    step = max(1, 512 // len(audio))
    embeddings = torch.cat([
        get_scene_embeddings(torch.cat([audio[:, start:start + window] for start in starts[i:i + step]]), model)
        for i in range(0, len(starts), step)
    ])
    embeddings = embeddings.view(len(starts), len(audio), -1).transpose(0, 1)
    timestamps = torch.arange(len(starts), device=embeddings.device) * 50.0
    return embeddings, timestamps.float().expand(len(audio), -1)


def run(args, model, transform, tokenizer):
    from heareval.embeddings.task_embeddings import Embedding, task_embeddings
    from heareval.predictions.runner import runner

    name = args.dataset.removeprefix(PREFIX)
    output = Path(args.output)
    done = output / ".done.embeddings"
    if not (args.skip_existing and done.exists()):
        task = Path(args.dataset_root) / name
        if not task.exists():
            download(name, args.dataset_root)
        for marker in (done, output / "prediction-done.json"):
            marker.unlink(missing_ok=True)
        embedding = Embedding(__name__, model_options={"model": model, "transform": transform})
        task_embeddings(embedding, task, output)
        for split in json.loads((output / "task_metadata.json").read_text())["splits"]:
            shutil.rmtree(output / split)   # heareval's per-file copies; its probes read only the per-split files
        done.touch()
    # heareval's downstream probes (skipped once prediction-done.json exists); it reloads checkpoints with pickle
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    runner.main([str(output)], standalone_mode=False)
