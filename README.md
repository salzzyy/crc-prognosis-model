# CRC Survival Prediction - MLOps Pipeline

<p align="center"> <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python Badge"/> <img src="https://img.shields.io/badge/MLOps-Kubeflow-informational" alt="Kubeflow Badge"/> <img src="https://img.shields.io/badge/MLflow-Tracking-orange" alt="MLflow Badge"/> <img src="https://img.shields.io/badge/Docker-Containerization-blue" alt="Docker Badge"/> <img src="https://img.shields.io/badge/Scikit--learn-Modeling-green" alt="Scikit-learn Badge"/> <img src="https://img.shields.io/badge/Built%20with-%E2%9D%A4-red" alt="Love Badge"/> </p>



**Use Case**
* This project is designed to predict Colorectal Cancer (CRC) patient survival using machine learning models.
It builds a complete MLOps pipeline that processes raw patient data, selects important features, trains a model, evaluates it, and tracks experiments with MLflow.

* The main objective is to streamline the end-to-end machine learning workflow — from data preprocessing to model training and evaluation — making it ready for production deployment.



**What's in this Project?**
* Custom Exception Handling (custom_exception.py):
    * Captures and logs detailed error messages with file names and line numbers for easier debugging.

* Data Processing Pipeline (data_processing.py):

    * Loads raw CSV data.

    * Preprocesses data (label encoding, scaling).

    * Feature selection using Chi-Squared tests.

    * Saves the processed data for model training.

* Model Training Pipeline (model_training.py):

    * Loads the processed data.

    * Trains a GradientBoostingClassifier.

    * Evaluates model performance (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

    * Tracks metrics using MLflow.

* Logging Module (logger.py):
    Saves logs with timestamps into a logs/ directory for easier traceability.

* MLOps Pipeline (Kubeflow Pipelines) (mlop_pipeline.py):
        Connects data processing and model training steps into one automated pipeline.

* Dockerfile:
            Containerizes the application for consistent deployment.





| Tool | Purpose |
|------|---------|
| Python 3.9 | Main programming language |
| Pandas, NumPy, Scikit-learn | Data processing and model building |
| MLflow | Experiment tracking |
| Joblib | Model and data serialization |
| Kubeflow Pipelines (KFP) | MLOps pipeline orchestration |
| Docker | Containerization |
| Logging | Tracking and debugging |
| Chi-Squared Feature Selection | Feature importance evaluation |


# Workflow

## Data Processing Stage:
- Loads raw data.
- Preprocesses it (drop unnecessary columns, encode categories).
- Performs feature selection (top 5 features).
- Scales numerical features.
- Splits into train/test sets and saves them.

## Model Training Stage:
- Loads preprocessed data.
- Trains a Gradient Boosting model.
- Evaluates on test data (accuracy, precision, recall, F1-score, ROC-AUC).
- Logs metrics to MLflow.

## Pipeline Automation:
- Using Kubeflow Pipelines, both stages are connected.
- Ensures that model training only happens after successful data processing.
- Managed inside a Docker container for reproducibility.





```text
├── src
│   ├── custom_exception.py
│   ├── data_processing.py
│   ├── logger.py
│   ├── model_training.py
├── artifacts
│   ├── raw
│   │   └── data.csv
│   ├── processed
│   │   └── (processed data files)
│   └── models
│       └── model.pkl
├── logs
│   └── (daily logs)
├── mlop_pipeline.py
├── Dockerfile
├── requirements.txt (optional if you create it)
└── README.md
```



1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```


2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the ML Pipeline Locally
```bash
python mlop_pipeline.py
```


🐳 Running with Docker (Optional)
#### Build Docker Image:
```bash
docker build -t crc-survival-pipeline .
```

#### Run Container:
```bash
docker run -p 5000:5000 crc-survival-pipeline
```
