import os
import sys
from src.Heart.logger import logging
from src.Heart.exception import customexception
from src.Heart.components.Data_ingestion import DataIngestion
from src.Heart.components.Data_transformation import DataTransformation
from src.Heart.components.Model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            ingestion = DataIngestion()
            test_data, train_data = ingestion.initiate_data_ingestion()
            transformation = DataTransformation()
            train_arr, test_arr = transformation.initiate_data_transformation(train_data, test_data)
            trainer = ModelTrainer()
            trainer.initiate_model_trainer(train_arr, test_arr)
        except Exception as e:
            raise customexception(e, sys)
