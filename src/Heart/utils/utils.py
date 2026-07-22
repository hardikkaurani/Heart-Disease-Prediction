import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.Heart.exception import customexception
from src.Heart.logger import logging
from sklearn.metrics import accuracy_score, r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info(f'Saving object to {file_path}')
        raise customexception(e, sys)

def load_object(file_path):
    try:
        logging.info(f'Loading object from {file_path}')
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise customexception(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            report[list(models.keys())[i]] = accuracy_score(y_test, y_test_pred)
        return report
    except Exception as e:
        raise customexception(e, sys)
