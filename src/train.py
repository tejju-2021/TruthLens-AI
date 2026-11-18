"""
TruthLens AI — Training Orchestration Entrypoint
Allows triggering model training runs (e.g. XGBoost baseline or DistilBERT fine-tuning).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_xgboost import main as train_xgboost_main


def main():
    print("TruthLens AI Model Training Suite")
    print("1. Train XGBoost Baseline (using 80/20 stratified split & 50k TF-IDF n-grams)")
    print("Executing XGBoost training...")
    train_xgboost_main()


if __name__ == "__main__":
    main()
