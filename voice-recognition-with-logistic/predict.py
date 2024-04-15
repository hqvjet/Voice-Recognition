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
    print("Classificationn Report: \n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("Model's accuracy: ", accuracy)
    return accuracy

def build_model():
    ds = load_dataset()
    x_data = ds[:, :-1]
    y_data = ds[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    return model, eval_model(y_test, y_pred)
    
def predict(x_test):
    with open('res/log_reg_model.pkl', 'rb') as file:
        model = pickle.load(file)

    return model.predict(x_test)[0]

# test data
ds = load_dataset()
x_data = ds[:, :-1]
x_test = x_data[0].reshape(1, -1)

def train_model(train_loop):
    max = 0.0
    model = None
    for i in range(0, train_loop):
        # BUILD MODEL
        cur_model, eccuracy = build_model()
        if (eccuracy >= max):
            model = cur_model
            max = eccuracy
    
    with open("res/log_reg_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(max)
         
# TRAIN MODEL
train_model(1000)

# PREDICT
# print(predict(x_test))