import os
import sys
from dataclasses import dataclass

from src.Heart.exception import customexception
from src.Heart.logger import logging
from src.Heart.utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
