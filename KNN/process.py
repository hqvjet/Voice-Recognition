import librosa
import numpy as np

def preprocessing(file_path):

    audio, sr = librosa.load(file_path, sr=8000, duration=3)

    if audio.shape[0] < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - audio.shape[0]), 'constant')

    return np.array(audio)
