import os
from KNN.constants import *
import librosa
import numpy as np
from classify import getlabel
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def getData(user):
    datasets = {
        'voice': [],
        'user': []
    }
    print('GETTING DATA FROM USER:', user)

    # Get the list of files in the user's directory
    audio_files = [file for file in os.listdir(PATH + user) if file.endswith('.m4a')]
    for audio_file in audio_files:
        print('GETTING DATA FROM AUDIO FILE:', audio_file)

        # Load the audio file
        audio, sr = librosa.load(PATH + user + '/' + audio_file, sr=8000, duration=3)
        print(audio.shape)

        datasets.get('voice').append(audio)
        datasets.get('user').append(getlabel(user))

    # PAD THE AUDIO dataset
    for i in range(len(datasets.get('voice'))):
        if datasets.get('voice')[i].shape[0] < MAX_LEN:
            datasets.get('voice')[i] = np.pad(datasets.get('voice')[i], (0, MAX_LEN - datasets.get('voice')[i].shape[0]), 'constant')

    return np.array(datasets.get('voice')), np.array(datasets.get('user'))

users = [user for user in os.listdir(PATH)]

voices = []
labels = []
for user in users:
    voice, label = getData(user)
    voices.append(voice)
    labels.append(label)

voices = np.array(voices)
labels = np.array(labels)

with open('res/voice.pickle', 'wb') as f:
    pickle.dump(voices, f)
with open('res/label.pickle', 'wb') as f:
    pickle.dump(labels, f)

