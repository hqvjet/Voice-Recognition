from fastapi import FastAPI, File, UploadFile
from typing import Union
import numpy as np
import pickle
from KNN.process import preprocessing 

app = FastAPI()

with open('res/knn.pickle', 'rb') as f:
    knn = pickle.load(f)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    contents = await file.read()

    with open('res/' + file.filename, 'wb') as audio_file:
        audio_file.write(contents)

    x = preprocessing('res/' + file.filename)

    y = knn.predict(x)

    return {'prediction:': y}

