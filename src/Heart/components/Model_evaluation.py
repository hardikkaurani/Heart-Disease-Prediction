import os
import sys
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.Heart.logger import logging
from src.Heart.exception import customexception
from src.Heart.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def initate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            model_path = os.path.join("Artifacts", "Model.pkl")
            model = load_object(model_path)

            predicted_qualities = model.predict(X_test)
            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predicted_qualities)
            logging.info(f"Evaluation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            try:
                import mlflow
                import mlflow.sklearn

                mlflow.set_registry_uri("https://dagshub.com/HemaKalyan45/Heart-Disease-Prediction.mlflow")
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                with mlflow.start_run():
                    mlflow.log_metric("Testing Accuracy", accuracy)
                    mlflow.log_metric("Precision Score", precision)
                    mlflow.log_metric("Recall Score", recall)
                    mlflow.log_metric("F1 Score", f1)

                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(model, "Model", registered_model_name="ml_model")
                    else:
                        mlflow.sklearn.log_model(model, "Model")
            except Exception as mlflow_err:
                logging.info(f"MLflow tracking skipped or unavailable: {mlflow_err}")

        except Exception as e:
            raise customexception(e, sys)




