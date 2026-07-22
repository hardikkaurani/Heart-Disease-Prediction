import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from src.Heart.exception import customexception
from src.Heart.logger import logging
from src.Heart.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return acc, precision, recall, f1
