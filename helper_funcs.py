# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 02:27:30 2020

@author: Shaurya-PC
"""


import numpy as np
import pandas as pd
from get_feature_matrix import extract_feature
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def create_and_save():
    dataset = pd.read_csv("final_features.csv")
    X = dataset.iloc[:, 2:-1].values
    Y = dataset.iloc[:, [-1]].values
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Splitting data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    gb = GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    acc = gb.score(X_test, Y_test)
    print(acc)
    saved_file = 'gradient_boosting.pkl'
    file = open(saved_file, 'wb')
    # dump information to that file
    pickle.dump(gb, file)
    # close the file
    file.close()

    return [saved_file, scaler]


def load(fname):
    # open a file, where you stored the pickled data
    file = open(fname, 'rb')
    # dump information to that file
    model = pickle.load(file)
    # close the file
    file.close()

    return model


def predict(fname, model, scaler):
    new_X = extract_feature(fname)

    new_X = new_X.iloc[:, 2:].values
    new_X = scaler.transform(new_X)

    pred = model.predict(new_X).flatten()[0]

    pred_values = {0: 'anger', 1: 'happy', 2: 'sad', 3: 'disgust', 4: 'fear', 5: 'surprise'}

    return pred_values[pred]
