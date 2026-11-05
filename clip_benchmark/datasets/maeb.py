"""MAEB (Massive Audio Embedding Benchmark): task selection and encoder interface; evaluation is done by MTEB."""

from pathlib import Path

import mteb
import numpy as np
import torch
from mteb._set_seed import _set_seed
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

PREFIX = "maeb/"


def _tasks():
    return {task.metadata.name: task for task in mteb.get_benchmark("MAEB(beta)").tasks}


def dataset_names():
    return [PREFIX + name for name in _tasks()]


def fold_all_caps(text):
    """Lowercase text written entirely in capitals (at least three words); leave all other text unchanged.
    The GigaSpeech queries are all-caps transcripts, which case-sensitive tokenizers split into rare pieces."""
    if not (text.isupper() and len(text.split()) >= 3):
        return text
    # isupper() is also true for an uncased script (Korean, Persian, ...) with a Latin acronym: fold only if
    # most letters belong to a cased script.
    letters = [c for c in text if c.isalpha()]
    cased = sum(c.isupper() or c.islower() for c in letters)
    return text.lower() if cased * 2 >= len(letters) else text


class MAEBEncoder(AbsEncoder):
    def __init__(self, model, transform, tokenizer, model_name, device):
        self.model, self.transform, self.tokenizer, self.device = model, transform, tokenizer, device
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": f"open_clap/{model_name}",
            "languages": ["eng-Latn"],
            "max_tokens": float("inf"),
            "open_weights": True,
            "framework": ["PyTorch"],
            "similarity_fn_name": ScoringFunction.COSINE,
            "use_instructions": False,
            "modalities": ["audio", "text"],
        })

    @torch.inference_mode()
    def encode(self, inputs, **kwargs):
        audio = "audio" in inputs.dataset.features
        text = "text" in inputs.dataset.features
        assert audio != text, "MAEB batches must contain exactly one input modality."
        if audio:
            inputs.collate_fn = AudioCollator(target_sampling_rate=self.model.audio.cfg.sample_rate)
        embeddings = []
        for batch in inputs:
            if audio:
                items = batch["audio"]
                features = self.model.encode_audio(
                    self.transform([item["array"] for item in items], [item["sampling_rate"] for item in items]),
                )
            else:
                texts = self.tokenizer([fold_all_caps(text) for text in batch["text"]])
                features = self.model.encode_text(texts.to(self.device))
            embeddings.append(features.float().cpu().numpy())
        return np.concatenate(embeddings)


def run(args, model, transform, tokenizer):
    """Evaluate the task `args.dataset` ("maeb/<task>"); data is downloaded by MTEB. `args.seed` seeds the
    probes and clustering (MTEB's own default is 42)."""
    task = _tasks()[args.dataset.removeprefix(PREFIX)]
    task.seed = args.seed
    task.rng_state, task.np_rng = _set_seed(args.seed)
    encoder = MAEBEncoder(model, transform, tokenizer, args.model, args.device)
    return mteb.evaluate(
        encoder,
        [task],
        cache=mteb.ResultCache(str(Path(args.output) / f"native_mteb_{mteb.__version__}")),
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="only-missing" if args.skip_existing else "always",
        raise_error=True,
    )
