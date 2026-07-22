from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.Heart.pipeline.Prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            pass
        except Exception as e:
            return render_template('error.html', error=str(e))
