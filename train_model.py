# train_model.py
import os
import json
import logging
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

log = logging.getLogger("train_model")
log.setLevel(logging.INFO)


AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", Path.home() / "airflow"))
DATA_DIR = AIRFLOW_HOME / "data"
ARTIFACTS_DIR = AIRFLOW_HOME / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TRANS_DIR = ARTIFACTS_DIR / "transformers"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def safe_ordinal_encoder():
    """
    Create an OrdinalEncoder compatible with different sklearn versions.
    If handle_unknown='use_encoded_value' is not supported, fallback to ignore.
    """
    try:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        enc = OrdinalEncoder(handle_unknown="ignore")
    return enc


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_col: str, tfidf_max_features: int = 1000):
    """
    Build a pipeline:
      - numeric -> StandardScaler
      - categorical -> OrdinalEncoder (we pre-collapsed rare categories)
      - text -> TfidfVectorizer
    Important: pass text_col as a string to ColumnTransformer so TfidfVectorizer receives a 1D sequence.
    """
    transformers = []

    if num_cols:
        num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols:
        cat_transformer = Pipeline(steps=[("ord", safe_ordinal_encoder())])
        transformers.append(("cat", cat_transformer, cat_cols))

    # Use TfidfVectorizer directly and pass text_col as a string (ColumnTransformer will pass a Series)
    text_transformer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
    transformers.append(("text", text_transformer, text_col))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,  # allow sparse output if TF-IDF dominates
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)),
        ]
    )
    return pipeline


def train(debug_sample: Optional[int] = None):
    """
    Train pipeline:
      - load cleaned CSV and metadata
      - optional debug_sample to limit dataset size for testing
      - GridSearchCV over a small grid
      - save best pipeline, preprocessor, metrics and params
    """
    clean_path = DATA_DIR / "wine_clean.csv"
    meta_path = ARTIFACTS_DIR / "columns.json"

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    df = pd.read_csv(clean_path)
    meta = json.loads(meta_path.read_text())

    target_col = meta.get("target", "points")
    num_cols = meta.get("num_columns", [])
    cat_cols = meta.get("cat_columns", [])
    text_col = meta.get("text_column", "description")

    # Keep only existing columns
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    if text_col not in df.columns:
        df[text_col] = ""

    # Optional debug sampling to reduce runtime on large dataset
    if debug_sample is not None:
        df = df.sample(n=min(debug_sample, len(df)), random_state=42).reset_index(drop=True)
        log.info("Using debug sample of size %d", len(df))

    # Ensure text column is string type and has no nulls
    df[text_col] = df[text_col].fillna("").astype(str)

    # Prepare X and y
    feature_cols = []
    feature_cols.extend(cat_cols)
    feature_cols.extend(num_cols)
    feature_cols.append(text_col)

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).values

    log.info("Data shapes before split: X=%s y=%s", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Quick sanity checks: ensure text column length matches rows
    if X_train.shape[0] != X_train[text_col].shape[0]:
        raise ValueError("Mismatch between X_train rows and text column length")

    pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols, text_col=text_col, tfidf_max_features=1000)

    param_grid = {
        "regressor__alpha": [0.0001, 0.001],
        "regressor__penalty": ["l2", "elasticnet"],
        "regressor__l1_ratio": [0.15, 0.5],
    }

    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=2, verbose=1, error_score="raise")
    log.info("Starting GridSearchCV with X_train=%s", X_train.shape)
    try:
        gs.fit(X_train, y_train)
    except Exception as e:
        log.exception("GridSearchCV failed: %s", e)
        # Re-raise so Airflow marks the task as failed and logs the full traceback
        raise

    log.info("GridSearchCV finished")

    best = gs.best_estimator_

    # Predictions and metrics
    y_pred = best.predict(X_val)
    rmse, mae, r2 = eval_metrics(y_val, y_pred)
    log.info("Validation metrics - RMSE: %.4f MAE: %.4f R2: %.4f", rmse, mae, r2)

    # Save full pipeline (includes preprocessor)
    model_path = MODELS_DIR / "best_pipeline.pkl"
    joblib.dump(best, model_path)
    log.info("Saved model pipeline to %s", model_path)

    # Save preprocessor separately (optional)
    try:
        preproc = best.named_steps.get("preprocessor")
        if preproc is not None:
            joblib.dump(preproc, TRANS_DIR / "preprocessor.pkl")
            log.info("Saved preprocessor to %s", TRANS_DIR / "preprocessor.pkl")
    except Exception as e:
        log.warning("Could not save preprocessor separately: %s", e)

    # Save metrics and best params
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS_DIR / "best_params.json").write_text(json.dumps(gs.best_params_, indent=2))

    # Save category mappings (useful for inference)
    cat_mappings = meta.get("cat_mappings", {})
    (TRANS_DIR / "cat_mappings.json").write_text(json.dumps(cat_mappings, indent=2))

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For quick local debug you can pass debug_sample to limit data size:
    # train(debug_sample=20000)
    train()
