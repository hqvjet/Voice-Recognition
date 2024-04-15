import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from KNN.constants import *
import librosa
import numpy as np
from classify import getlabel
import pickle
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
        audio, sr = librosa.load(PATH + user + '/' + audio_file, sr=16000, duration=3)
        print(audio.shape)

        datasets.append(audio)

    # PAD THE AUDIO dataset
    for i in range(len(datasets)):
        if datasets[i].shape[0] < MAX_LEN:
            datasets[i] = np.pad(datasets[i], (0, MAX_LEN - datasets[i].shape[0]), 'constant')
        datasets[i] = np.append(datasets[i], getlabel(user))
    print(datasets[i].shape)

    return np.array(datasets)

def preprocessing():

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

def train():
    with open('res/voice.pickle', 'rb') as f:
        voices = pickle.load(f)

    X_train = voices[:, :-1]
    y_train = voices[:, -1]
    
    X_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(x_test)
    
    k_values = range(1, 15)
    opt_k = 1
    max_acc = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(x_test)
        print(pred)
        acc = accuracy_score(y_test, pred)
        print(acc)
        if acc > max_acc:
            max_acc = acc
            opt_k = k
            with open('res/knn.pickle', 'wb') as f:
                pickle.dump(knn, f)

    print('Model saved')

    with open('res/knn.pickle', 'rb') as f:
        knn = pickle.load(f)
    y_pred = knn.predict(x_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

preprocessing()
train()
