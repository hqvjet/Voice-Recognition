import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def load_dataset():
    with open("res/voice.pickle", "rb") as f:
        return pickle.load(f)

def eval_model(y_test, y_pred): 
    accuracy = accuracy_score(y_test, y_pred)
    print("Model's accuracy: ", accuracy)

def build_model():
    ds = load_dataset()
    x_data = ds[:, :-1]
    y_data = ds[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    eval_model(y_test, y_pred)

    with open("res/log_reg_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
def predict(x_test):
    with open('res/log_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)

    return model.predict(x_test)[0]

# test data
ds = load_dataset()
x_data = ds[:, :-1]
x_test = x_data[0].reshape(1, -1)

# BUILD MODEL
# build_model()

# PREDICT
predict(x_test)