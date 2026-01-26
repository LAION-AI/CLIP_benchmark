import io
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None

# CLAP uses 10 seconds at 48kHz = 480000 samples
TARGET_LENGTH = 480000

def audio_decoder(key, data):
    """
    Decodes audio data using torchaudio.
    Supports wav, flac, mp3 extensions.
    """
    extension = key.split(".")[-1].lower()
    if extension in ["wav", "flac", "mp3"]:
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio decoding")
        
        # Load audio from bytes
        wav, sr = torchaudio.load(io.BytesIO(data))
        
        # Resample to 48kHz if needed (CLAP default)
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
        
        # Convert to mono and squeeze to 1D for CLAP compatibility
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        
        # Pad or truncate to fixed length
        if wav.shape[0] < TARGET_LENGTH:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LENGTH - wav.shape[0]))
        else:
            wav = wav[:TARGET_LENGTH]
            
        return wav
    return None
