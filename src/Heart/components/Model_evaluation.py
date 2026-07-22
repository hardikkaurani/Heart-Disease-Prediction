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
