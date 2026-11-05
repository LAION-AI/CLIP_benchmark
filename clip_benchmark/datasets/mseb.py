"""MSEB (Massive Sound Embedding Benchmark): runs the upstream evaluation of one task with a CLAP model."""

import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from mseb import encoder, types

PREFIX = "mseb/"


class MSEBEncoder(encoder.MultiModalEncoder):
    def __init__(self, model, transform, tokenizer, device):
        super().__init__()
        self.model, self.transform, self.tokenizer, self.device = model, transform, tokenizer, device

    def _setup(self):
        pass

    def _check_input_types(self, batch):
        assert all(isinstance(item, (types.Sound, types.Text)) for item in batch), "Expected Sound or Text inputs."

    @torch.inference_mode()
    def _encode(self, batch) -> Sequence[types.MultiModalEmbedding]:
        output = [None] * len(batch)
        for modality in (types.Sound, types.Text):
            indices = [i for i, item in enumerate(batch) if isinstance(item, modality)]
            if not indices:
                continue
            items = [batch[i] for i in indices]
            if modality is types.Sound:
                sounds = [encoder.resample_sound(item, item.context.sample_rate) for item in items]
                features = self.model.encode_audio(self.transform(
                    [item.waveform for item in sounds], [item.context.sample_rate for item in sounds],
                ))
            else:
                # MSEB passes raw label names (FSD50K: "Accelerating_and_revving_and_vroom").
                texts = self.tokenizer([item.text.replace("_", " ") for item in items])
                features = self.model.encode_text(texts.to(self.device))
            for i, item, feature in zip(indices, items, features.float().cpu().numpy(), strict=True):
                if modality is types.Sound:
                    output[i] = types.SoundEmbedding(
                        embedding=feature[None], context=item.context,
                        timestamps=np.array([[0., item.context.length / item.context.sample_rate]]),
                    )
                else:
                    output[i] = types.TextEmbedding(
                        embedding=feature[None], context=item.context,
                        spans=np.array([[0, len(item.text)]]),
                    )
        return output


def dataset_names():
    """The English sound and text embedding tasks of MSEB."""
    birdset, locales = ("HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH", "XCL", "XCM"), ("EnAu", "EnGb", "EnIn", "EnPh", "EnUs")
    svq = ("ContinuousStability", "DocumentInLangRetrievalSmallIndex", "PassageInLangRetrieval", "QueryReranking",
           "SpanInLangReasoning", "SpeakerGenderClassification")
    return [PREFIX + name for name in ["FSD50KTestClassification", "FSD50KTestClustering"]
            + [f"Birdset{s}Classification" for s in birdset] + [f"BirdsetClustering{s}" for s in birdset]
            + [f"SVQClustering{l}" for l in locales] + [f"SVQ{l}{t}" for l in locales for t in svq]]


def data_dir(root, name):
    """MSEB reads each source dataset from one directory (SVQ: the 2.0 layout of google/svq); it downloads FSD50K itself."""
    source = next(s for s in ("FSD50K", "Birdset", "SVQ") if name.startswith(s))
    return str(Path(root) / {"FSD50K": "fsd50k", "Birdset": "birdset", "SVQ": "svq/2.0.0"}[source])


def run(args, model, transform, tokenizer):
    from absl import flags
    from mseb import leaderboard, runner, task, tasks

    name = args.dataset.removeprefix(PREFIX)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{name}.jsonl"
    if args.skip_existing and result_path.exists():
        return
    cache = output / "cache" / name
    if not args.skip_existing:
        result_path.unlink(missing_ok=True)
        if cache.exists():
            shutil.rmtree(cache)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["clip_benchmark"], known_only=True)
    basepath = data_dir(args.dataset_root, name)
    if name.startswith("Birdset"):  # the BirdSet files MSEB reads, made from BirdSet's test_5s data
        snapshot_download("laion/eval-mseb-birdset", repo_type="dataset", revision="7f26b0b89b6c17d74ec580e06a5eec5e8a966ded",
                          local_dir=basepath)
    if name.startswith("SVQ"):  # MSEB reads SVQ from disk: the English part of SVQ 2.0
        snapshot_download("google/svq", repo_type="dataset", revision="74f17ec92f0654860c160029d3542199626f63ef",
                          local_dir=Path(basepath).parent, allow_patterns=[
                              "2.0.0/utt_index.jsonl", "2.0.0/utts_en_*.parquet", "2.0.0/*_in_lang*index.parquet"])
    flags.FLAGS["dataset_basepath"].value = basepath
    flags.FLAGS[task.TASK_CACHE_BASEPATH.name].value = str(cache / "tasks")
    evaluator = runner.DirectRunner(
        encoder=MSEBEncoder(model, transform, tokenizer, args.device), batch_size=args.batch_size,
        num_threads=1, output_path=str(cache / "embeddings"),
    )
    benchmark_task = tasks.get_task_by_name(name)()
    benchmark_task.setup(runner=evaluator)
    results = leaderboard.run_benchmark(
        encoder_name=args.model, runner=evaluator, task=benchmark_task,
        base_model=args.model_type, tags=[args.pretrained],
    )
    temporary = result_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(result.to_json() + "\n" for result in results))
    temporary.replace(result_path)
