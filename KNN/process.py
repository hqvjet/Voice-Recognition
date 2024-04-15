import librosa
import numpy as np
from KNN.constants import * 

def preprocessing(file_path):
    print(file_path)
    audio, sr = librosa.load(file_path, sr=16000, duration=3)

    if audio.shape[0] < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - audio.shape[0]), 'constant')

    return np.array(audio)
