import os
from KNN.constants import *
import librosa
import numpy as np
from classify import getlabel
import pickle
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def getData(user):
    datasets = []
    print('GETTING DATA FROM USER:', user)

    # Get the list of files in the user's directory
    audio_files = [file for file in os.listdir(PATH + user) if file.endswith('.m4a')]
    for audio_file in audio_files:
        print('GETTING DATA FROM AUDIO FILE:', audio_file)

        # Load the audio file
        audio, sr = librosa.load(PATH + user + '/' + audio_file, sr=8000, duration=3)
        print(audio.shape)

        datasets.append(audio)

    # PAD THE AUDIO dataset
    for i in range(len(datasets)):
        if datasets[i].shape[0] < MAX_LEN:
            datasets[i] = np.pad(datasets[i], (0, MAX_LEN - datasets[i].shape[0]), 'constant')
        datasets[i] = np.append(datasets[i], getlabel(user))
        print(datasets[i].shape)

    return np.array(datasets)

users = [user for user in os.listdir(PATH)]
voices = []

for user in users:
    voice = getData(user)
    for i in voice:
        voices.append(i)

voices = np.array(voices)
np.random.shuffle(voices)
print(voices.shape)

with open('res/voice.pickle', 'wb') as f:
    pickle.dump(voices, f)

