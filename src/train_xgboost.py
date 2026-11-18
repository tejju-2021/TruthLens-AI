"""
TruthLens AI — XGBoost Classical ML Baseline Training & Evaluation
Uses cleaned ISOT dataset, identical 80/20 stratified split, TF-IDF features (max_features=50000, (1,2) n-grams).
Fits feature transformer only on training data.
Evaluates accuracy, precision, recall, F1, and confusion matrix.
Saves model and vectorizer to models/ and updates evaluation/baseline_results.csv.
"""

import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "truthlens_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
EVALUATION_DIR = ROOT_DIR / "evaluation"
BASELINE_CSV = EVALUATION_DIR / "baseline_results.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    X = df["content"].fillna("")
    y = df["label"].astype(int)

    # Identical stratified 80/20 split as EDA baseline
    print("\nSplitting train/test (80/20 stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # Fit TF-IDF only on training data
    print("\nExtracting TF-IDF features (max_features=50000, ngram_range=(1,2), min_df=2)...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    t0 = time.time()
    X_train_tfidf = tfidf.fit_transform(X_train)
    fit_transform_time = time.time() - t0
    print(f"TF-IDF fit_transform completed in {fit_transform_time:.2f}s")
    print(f"Train TF-IDF shape: {X_train_tfidf.shape} (sparse CSR matrix)")

    X_test_tfidf = tfidf.transform(X_test)
    print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

    # Save TF-IDF Vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf, vectorizer_path)
    print(f"Saved TF-IDF vectorizer to {vectorizer_path}")

    # Initialize XGBoost Classifier with histogram tree method for efficient sparse CSR matrix handling
    print("\nTraining XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    train_start = time.time()
    xgb_model.fit(X_train_tfidf, y_train)
    training_time = time.time() - train_start
    print(f"XGBoost training completed in {training_time:.2f}s")

    # Save trained XGBoost model
    model_path = MODELS_DIR / "xgboost_model.joblib"
    joblib.dump(xgb_model, model_path)
    print(f"Saved XGBoost model to {model_path}")

    # Inference & Evaluation
    print("\nEvaluating on test set...")
    inf_start = time.time()
    y_pred = xgb_model.predict(X_test_tfidf)
    y_proba = xgb_model.predict_proba(X_test_tfidf)
    inference_time = time.time() - inf_start
    latency_per_sample_ms = (inference_time / len(X_test)) * 1000

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n--- XGBOOST TEST EVALUATION RESULTS ---")
    print(f"Accuracy:  {acc:.6f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.6f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.6f} ({rec*100:.2f}%)")
    print(f"F1 Score:  {f1:.6f} ({f1*100:.2f}%)")
    print(f"Inference Latency: {latency_per_sample_ms:.4f} ms/sample")
    print(f"\nConfusion Matrix (tn, fp, fn, tp):")
    print(cm)
    print("\nClassification Report:")
    report_dict = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # Save detailed evaluation JSON
    eval_record = {
        "model": "TF-IDF + XGBoost",
        "input": "Title + Text",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "training_time_seconds": round(training_time, 2),
        "inference_latency_per_sample_ms": round(latency_per_sample_ms, 4),
        "classification_report": report_dict,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist"
        }
    }

    eval_json_path = EVALUATION_DIR / "xgboost_evaluation.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)
    print(f"Saved evaluation details to {eval_json_path}")

    # Update evaluation/baseline_results.csv
    if BASELINE_CSV.exists():
        baseline_df = pd.read_csv(BASELINE_CSV)
    else:
        baseline_df = pd.DataFrame(columns=["Model", "Input", "Accuracy", "Precision", "Recall", "F1 Score"])

    # Check if XGBoost already in baseline_df, update or append
    xgb_row = {
        "Model": "TF-IDF + XGBoost",
        "Input": "Title + Text",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    # Filter out any prior XGBoost entry
    baseline_df = baseline_df[baseline_df["Model"] != "TF-IDF + XGBoost"]
    baseline_df = pd.concat([baseline_df, pd.DataFrame([xgb_row])], ignore_index=True)
    baseline_df.to_csv(BASELINE_CSV, index=False)
    print(f"Updated {BASELINE_CSV}:\n{baseline_df}")


if __name__ == "__main__":
    main()
