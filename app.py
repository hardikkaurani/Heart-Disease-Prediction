from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.Heart.pipeline.Prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')
