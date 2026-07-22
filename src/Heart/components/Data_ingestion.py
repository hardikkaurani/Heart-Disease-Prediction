import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.Heart.logger import logging
from src.Heart.exception import customexception

class DataIngestionConfig:
    raw_data_path:str = os.path.join("Artifacts","raw_data.csv")
    train_data_path:str = os.path.join("Artifacts","train_data.csv")
    test_data_path:str = os.path.join("Artifacts","test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            data = pd.read_csv(os.path.join("Notebook_Experiments", "Data", "heart.csv"))
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
