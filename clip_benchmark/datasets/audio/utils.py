import numpy as np

def pad_or_crop(audio_data, target_len):
    """
    Pads or crops the audio data to the target length.
    
    Args:
        audio_data (numpy.ndarray): The audio data to pad or crop.
        target_len (int): The target length of the audio data.
    
    Returns:
        numpy.ndarray: The padded or cropped audio data.
    """

    if len(audio_data) < target_len:
        padding = target_len - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')
    else:
        audio_data = audio_data[:target_len]
    return audio_data