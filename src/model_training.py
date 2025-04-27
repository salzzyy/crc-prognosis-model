import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn

# Load environment variables
load_dotenv()

# Configure MLflow to use DagsHub
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'https://dagshub.com/your-username/your-repo-name.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'your-username')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'your-token')

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info("Model Training Initialization...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            # Log dataset info
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("test_size", len(self.X_test))

            logger.info("Data loaded for Model")
        except Exception as e:
            logger.error(f"Error while loading data for model {e}")
            raise CustomException("Failed to load data for model..")
        
    def train_model(self):
        try:
            # Log model parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 3)
            
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(self.X_train, self.y_train)

            # Save model locally
            model_path = os.path.join(self.model_dir, "model.pkl")
            joblib.dump(self.model, model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(self.model, "gradient_boosting_model")
            
            # Log model artifact
            mlflow.log_artifact(model_path)

            logger.info("Model trained and saved successfully...")

        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model...")
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)

            y_proba = self.model.predict_proba(self.X_test)[:, 1] if len(self.y_test.unique()) == 2 else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            logger.info(f"Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}; F1-Score: {f1}")

            if y_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_proba)
                mlflow.log_metric("roc_auc", roc_auc)
                logger.info(f"ROC-AUC Score: {roc_auc}")

            # Create and log confusion matrix plot (optional)
            try:
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                cm = confusion_matrix(self.y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                
                # Save and log the figure
                cm_path = os.path.join(self.model_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path)
            except ImportError:
                logger.warning("Could not create confusion matrix plot - missing dependencies")

            logger.info("Model evaluation done...")

        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model...")
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    with mlflow.start_run(run_name="GradientBoostingClassifier"):
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        trainer = ModelTraining()
        trainer.run()