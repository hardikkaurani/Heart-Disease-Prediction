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
