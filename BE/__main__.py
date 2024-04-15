from fastapi import FastAPI, File, UploadFile
from typing import Union
import numpy as np
import pickle
from KNN.process import preprocessing 

app = FastAPI()

with open('res/log_reg_model.pkl', 'rb') as f:
    knn = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    with open('res/' + file.filename, 'wb') as audio_file:
        audio_file.write(contents)

    x = preprocessing('res/' + file.filename)

    y = knn.predict(np.array([x]))

    return {'prediction:': y.tolist()}

