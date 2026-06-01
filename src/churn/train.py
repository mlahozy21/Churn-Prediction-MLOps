"""Train the churn model, evaluate it, log to MLflow and save the pipeline."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split

from .data import FEATURES, TARGET, load_data
from .pipeline import build_pipeline

MODEL_PATH = "models/churn_pipeline.joblib"
METRICS_PATH = "models/metrics.json"


def train(csv_path: str | None = None, n_synthetic: int = 5000,
          test_size: float = 0.2, seed: int = 42, use_mlflow: bool = True):
    df = load_data(csv_path, n_synthetic)
    X = df[FEATURES]
    y = (df[TARGET] == "Yes").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    pipe = build_pipeline(random_state=seed)
    # cross-validated AUC on the training set
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "test_auc": float(roc_auc_score(y_te, proba)),
        "test_f1": float(f1_score(y_te, pred)),
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("churn-prediction")
            with mlflow.start_run():
                mlflow.log_params({"model": "HistGradientBoosting",
                                   "learning_rate": 0.05, "max_iter": 400,
                                   "n_synthetic": n_synthetic, "seed": seed})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(MODEL_PATH)
        except Exception as e:  # noqa: BLE001
            print(f"(MLflow logging skipped: {e})")

    print(f"CV AUC: {metrics['cv_auc_mean']:.3f} +/- {metrics['cv_auc_std']:.3f} | "
          f"test AUC: {metrics['test_auc']:.3f} | F1: {metrics['test_f1']:.3f} | "
          f"acc: {metrics['test_accuracy']:.3f}")
    print(f"Saved model to {MODEL_PATH}")
    return pipe, metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the churn model.")
    ap.add_argument("--csv", default=None, help="Path to the Telco CSV (optional).")
    ap.add_argument("--n-synthetic", type=int, default=5000)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    train(args.csv, args.n_synthetic, use_mlflow=not args.no_mlflow)
