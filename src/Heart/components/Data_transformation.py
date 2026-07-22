from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.Heart.logger import logging
from src.Heart.exception import customexception

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            num_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            preprocessor = ColumnTransformer([('num_pipeline', num_pipeline, num_cols)])
            return preprocessor
        except Exception as e:
            raise customexception(e, sys)
