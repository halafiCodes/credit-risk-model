import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")


def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "../data/processed/merged_dataset.csv")
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df


def load_feature_pipeline():
    base_dir = os.path.dirname(__file__)
    pipeline_path = os.path.join(base_dir, "../models/feature_pipeline.pkl")
    print(f"[INFO] Loading feature pipeline from {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    return pipeline


def preprocess_data(df, feature_pipeline):
    y = df['is_high_risk']
    X = df.drop(columns=['is_high_risk', 'CustomerId'])  # Keep all features for pipeline
    X_transformed = feature_pipeline.transform(X)
    return X_transformed, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    return metrics


def main():
    df = load_data()
    feature_pipeline = load_feature_pipeline()
    X, y = preprocess_data(df, feature_pipeline)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("Credit_Risk_Modeling")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Decide which model is "best" (here we choose RandomForest)
    best_model_name = "RandomForest"
    best_model = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"[INFO] Training {model_name}...")
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)
            print(f"[INFO] {model_name} metrics: {metrics}")

            for key, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(model, model_name)
            print(f"[INFO] {model_name} saved in MLflow.")

            # Save the best model for FastAPI
            if model_name == best_model_name:
                best_model = model

    # Save best model to file
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"[INFO] Best model saved as {best_model_path}")


if __name__ == "__main__":
    main()
