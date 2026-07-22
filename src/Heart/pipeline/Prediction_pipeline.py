import sys
import os
import pandas as pd
from src.Heart.exception import customexception
from src.Heart.logger import logging
from src.Heart.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("Artifacts", "model.pkl")
            preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self, age:int, sex:int, cp:int, trestbps:int, chol:int, fbs:int, restecg:int, thalach:int, exang:int, oldpeak:float, slope:int, ca:int, thal:int):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal
