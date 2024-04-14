import os
from constants import *
import librosa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

users = [user for user in os.listdir(PATH)]

for user in users:
    print('GETTING DATA FROM USER:', user)

    # Get the list of files in the user's directory
    audio_files = [file for file in os.listdir(PATH + user) if file.endswith('.m4a')]
    for audio_file in audio_files:
        print('GETTING DATA FROM AUDIO FILE:', audio_file)

        # Load the audio file
        audio, sr = librosa.load(PATH + user + '/' + audio_file, sr=8000, duration=3)
        print(audio.shape)
