"""
CLAP model loader for old LAION-CLAP pretrained checkpoints (lukewys/laion_clap).

Loads old checkpoints into the new open_clip CLAP architecture by remapping
state_dict keys. Uses open_clip — no dependency on the old laion_clap package.

Supported pretrained names (from lukewys/laion_clap on HuggingFace):
  630k-best, 630k-fusion-best, 630k-audioset-best, 630k-audioset-fusion-best,
  music_audioset_epoch_15_esc_90.14, music_speech_audioset_epoch_15_esc_89.98,
  music_speech_epoch_15_esc_89.25

Usage:
  load_clap(model_name="CLAP-HTSAT-tiny-Roberta-base", pretrained="630k-audioset-best")
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import librosa

import open_clip

logger = logging.getLogger(__name__)

# HuggingFace repo for old LAION-CLAP checkpoints
_HF_REPO = "lukewys/laion_clap"

# Known checkpoint names -> HF filenames
_CHECKPOINT_NAMES = {
    "630k-best": "630k-best.pt",
    "630k-fusion-best": "630k-fusion-best.pt",
    "630k-audioset-best": "630k-audioset-best.pt",
    "630k-audioset-fusion-best": "630k-audioset-fusion-best.pt",
    "music_audioset_epoch_15_esc_90.14": "music_audioset_epoch_15_esc_90.14.pt",
    "music_speech_audioset_epoch_15_esc_89.98": "music_speech_audioset_epoch_15_esc_89.98.pt",
    "music_speech_epoch_15_esc_89.25": "music_speech_epoch_15_esc_89.25.pt",
}


def _remap_old_state_dict(old_sd):
    """Remap old laion_clap state_dict keys to new open_clip CLAP keys.

    Key mapping:
      audio_branch.*      -> audio.encoder.*
      audio_projection.*  -> audio.proj.*
      text_branch.*       -> text.transformer.*
      text_projection.*   -> text.proj.*
      logit_scale_a       -> logit_scale
      audio_transform.*, text_transform.*, logit_scale_t -> dropped
    """
    new_sd = {}
    for k, v in old_sd.items():
        # Skip audio/text transforms (not used in new architecture)
        if "audio_transform" in k or "text_transform" in k:
            continue

        new_k = k
        new_k = new_k.replace("audio_branch.", "audio.encoder.")
        new_k = new_k.replace("audio_projection.", "audio.proj.")
        new_k = new_k.replace("text_branch.", "text.transformer.")
        new_k = new_k.replace("text_projection.", "text.proj.")

        new_sd[new_k] = v

    # Map logit_scale_a -> logit_scale (new model uses unified scale)
    if "logit_scale_a" in new_sd:
        new_sd["logit_scale"] = new_sd.pop("logit_scale_a")
    # Drop logit_scale_t (not in new model)
    new_sd.pop("logit_scale_t", None)

    return new_sd


def _fix_text_proj(model, state_dict):
    """Replace model.text.proj to match old checkpoint dimensions.

    Old laion_clap: Linear(768->512, bias) + ReLU + Linear(512->512, bias)
    New open_clip:  Linear(768->640, no bias) + GELU + Linear(640->512, no bias)

    We replace the new proj with one that matches the old checkpoint's shapes
    so that load_state_dict succeeds.
    """
    w0 = state_dict.get("text.proj.0.weight")
    if w0 is None:
        return
    out_dim_0, in_dim = w0.shape           # [512, 768]
    has_bias = "text.proj.0.bias" in state_dict

    w2 = state_dict.get("text.proj.2.weight")
    out_dim_2, mid_dim = w2.shape           # [512, 512]

    model.text.proj = nn.Sequential(
        nn.Linear(in_dim, out_dim_0, bias=has_bias),   # 768 -> 512
        nn.ReLU(),                                      # old model uses ReLU
        nn.Linear(mid_dim, out_dim_2, bias=has_bias),  # 512 -> 512
    )


def _detect_fusion(state_dict):
    """Detect if checkpoint uses fusion (has fusion_model / mel_conv2d keys)."""
    for k in state_dict:
        if "fusion_model" in k or "mel_conv2d" in k:
            return True
    return False


def load_clap(model_name: str = "CLAP-HTSAT-tiny-Roberta-base", pretrained: str = "", device="cpu", **kwargs):
    from . import ModelBundle

    # Load checkpoint first to detect fusion before creating model
    state_dict = None
    needs_fusion = False
    is_old_format = False

    if pretrained:
        if pretrained in _CHECKPOINT_NAMES:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(_HF_REPO, filename=_CHECKPOINT_NAMES[pretrained])
            logger.info(f"Downloaded {pretrained} from {_HF_REPO}")
        else:
            ckpt_path = pretrained

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        is_old_format = any(k.startswith("audio_branch.") for k in state_dict)
        needs_fusion = _detect_fusion(state_dict)

        if is_old_format:
            logger.info("Detected old laion_clap checkpoint, remapping keys...")
            state_dict = _remap_old_state_dict(state_dict)

    # Create model — use fusion-enabled config if checkpoint requires it
    if needs_fusion:
        from open_clip.model import CLAP
        cfg = open_clip.get_model_config(model_name)
        cfg["audio_cfg"]["enable_fusion"] = True
        cfg["audio_cfg"]["fusion_type"] = "aff_2d"
        model = CLAP(**cfg, output_dict=True)
        logger.info("Created model with enable_fusion=True (detected from checkpoint)")
    else:
        model = open_clip.create_model(model_name, output_dict=True)

    if state_dict is not None:
        if is_old_format:
            _fix_text_proj(model, state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info(f"Loaded checkpoint: {pretrained}")

    model = model.to(device)
    model.eval()

    model_cfg = open_clip.get_model_config(model_name) or {}
    audio_cfg = model_cfg.get("audio_cfg", {})

    clap_wrapper = CLAPWrapper(model)
    clap_tokenizer = open_clip.get_tokenizer(model_name)
    clap_transform = CLAPTransform()
    if needs_fusion:
        clap_loader = FusionAudioLoader(audio_cfg)
    else:
        clap_loader = AudioLoader(audio_cfg)

    return ModelBundle(
        model=clap_wrapper,
        transform=clap_transform,
        tokenizer=clap_tokenizer,
        audio_loader=clap_loader,
    )


class CLAPWrapper(torch.nn.Module):
    """Wrapper exposing encode_text / encode_audio for clip_benchmark."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_text(self, text, normalize=False):
        return self.model.encode_text(text, normalize=normalize)

    def encode_audio(self, audio, normalize=False):
        if isinstance(audio, list):
            audio = self._collate_audio_dicts(audio)
        return self.model.encode_audio(audio, normalize=normalize)

    @staticmethod
    def _collate_audio_dicts(audio_list):
        """Collate a list of per-sample audio dicts into a batched dict."""
        batch = {}
        for k in audio_list[0]:
            vals = [d[k] for d in audio_list]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals)
            else:
                batch[k] = torch.tensor(vals)
        return batch


class CLAPTransform:
    def __call__(self, audio):
        return audio


class AudioLoader:
    """Load audio from raw bytes, resample to target SR, pad/truncate to clip_samples."""

    def __init__(self, audio_cfg):
        self.sample_rate = audio_cfg.get("sample_rate", 48000)
        self.clip_samples = audio_cfg.get("clip_samples", 480000)

    def __call__(self, key, data):
        extension = key.split(".")[-1].lower()
        if extension not in ["wav", "flac", "mp3"]:
            return None

        try:
            audio_waveform, _ = librosa.load(io.BytesIO(data), sr=self.sample_rate)
            audio_waveform = torch.from_numpy(audio_waveform).float()

            longer = len(audio_waveform) > self.clip_samples
            if len(audio_waveform) < self.clip_samples:
                audio_waveform = F.pad(audio_waveform, (0, self.clip_samples - len(audio_waveform)))
            else:
                audio_waveform = audio_waveform[:self.clip_samples]

            return {"waveform": audio_waveform, "longer": torch.tensor(longer)}
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return None


class FusionAudioLoader:
    """Load audio for fusion models: produces mel_fusion (4-channel mel spectrogram).

    Fusion mode expects:
      mel_fusion: (4, T_frames, mel_bins) — global + 3 local mel chunks
      longer: bool — whether original audio > clip_samples
    """

    def __init__(self, audio_cfg):
        self.sample_rate = audio_cfg.get("sample_rate", 48000)
        self.clip_samples = audio_cfg.get("clip_samples", 480000)
        self.hop_size = audio_cfg.get("hop_size", 480)
        self.mel_bins = audio_cfg.get("mel_bins", 64)
        self.window_size = audio_cfg.get("window_size", 1024)
        self.fmin = audio_cfg.get("fmin", 50)
        self.fmax = audio_cfg.get("fmax", 14000)

    def _get_mel(self, waveform):
        """Compute log-mel spectrogram from waveform."""
        import torchaudio
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.window_size,
            win_length=self.window_size,
            hop_length=self.hop_size,
            center=True, pad_mode="reflect", power=2.0, norm=None, onesided=True,
            n_mels=self.mel_bins,
            f_min=self.fmin,
            f_max=self.fmax,
        )
        mel = mel_tf(waveform)
        mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
        return mel.T  # (T_frames, mel_bins)

    def __call__(self, key, data):
        extension = key.split(".")[-1].lower()
        if extension not in ["wav", "flac", "mp3"]:
            return None

        try:
            audio_waveform, _ = librosa.load(io.BytesIO(data), sr=self.sample_rate)
            audio_waveform = torch.from_numpy(audio_waveform).float()

            longer = len(audio_waveform) > self.clip_samples
            chunk_frames = self.clip_samples // self.hop_size + 1

            if longer:
                # Compute mel from full waveform, then create 4-channel fusion
                mel = self._get_mel(audio_waveform)
                total_frames = mel.shape[0]

                if chunk_frames >= total_frames:
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    longer = False
                else:
                    import numpy as np
                    ranges = np.array_split(
                        list(range(0, total_frames - chunk_frames + 1)), 3
                    )
                    if len(ranges[1]) == 0:
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        ranges[2] = [0]
                    # Deterministic: take middle of each range for eval
                    idx_front = ranges[0][len(ranges[0]) // 2]
                    idx_middle = ranges[1][len(ranges[1]) // 2]
                    idx_back = ranges[2][len(ranges[2]) // 2]

                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames]

                    # Global view: resize full mel to chunk_frames
                    import torchvision.transforms
                    mel_shrink = torchvision.transforms.Resize(
                        size=[chunk_frames, self.mel_bins]
                    )(mel[None])[0]

                    mel_fusion = torch.stack(
                        [mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back],
                        dim=0,
                    )

                # Also crop waveform for compatibility
                audio_waveform = audio_waveform[:self.clip_samples]
            else:
                # Short audio: pad waveform, 4 identical mel channels
                if len(audio_waveform) < self.clip_samples:
                    audio_waveform = F.pad(
                        audio_waveform, (0, self.clip_samples - len(audio_waveform))
                    )
                mel = self._get_mel(audio_waveform)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)

            return {
                "waveform": audio_waveform,
                "mel_fusion": mel_fusion,
                "longer": torch.tensor(longer),
            }
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return None
