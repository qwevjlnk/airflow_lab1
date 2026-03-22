# airflow_pipe.py
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Paths
AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", Path.home() / "airflow"))
DATA_DIR = AIRFLOW_HOME / "data"
ARTIFACTS_DIR = AIRFLOW_HOME / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TRANSFORMERS_DIR = ARTIFACTS_DIR / "transformers"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("airflow_pipe")
log.setLevel(logging.INFO)


def download_data():
    src = DATA_DIR / "winemag-data-130k-v2.csv"
    dst = DATA_DIR / "wine_raw.csv"
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")
    df = pd.read_csv(src)
    df.to_csv(dst, index=False)
    log.info("download_data: saved wine_raw.csv rows=%d", df.shape[0])
    return True


def clear_data():
    raw_path = DATA_DIR / "wine_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["points"])

    if "price" in df.columns:
        df["price_missing"] = df["price"].isna().astype(int)
        median_price = df["price"].median()
        df["price"] = df["price"].fillna(median_price)
    else:
        df["price"] = 0.0
        df["price_missing"] = 1

    df["description"] = df["description"].fillna("")
    df["desc_len"] = df["description"].str.len()

    candidate_cat = ["country", "province", "region_1", "variety", "winery"]
    cat_columns = [c for c in candidate_cat if c in df.columns]

    rare_threshold = 50
    cat_mappings = {}
    for c in cat_columns:
        df[c] = df[c].fillna("UNKNOWN")
        freqs = df[c].value_counts()
        rare_vals = freqs[freqs < rare_threshold].index.tolist()
        if rare_vals:
            df.loc[df[c].isin(rare_vals), c] = "OTHER"
        cat_mappings[c] = df[c].unique().tolist()

    clean_path = DATA_DIR / "wine_clean.csv"
    df.to_csv(clean_path, index=False)

    meta = {
        "target": "points",
        "num_columns": ["price", "price_missing", "desc_len"],
        "cat_columns": cat_columns,
        "text_column": "description",
        "cat_mappings": cat_mappings,
    }
    meta_path = ARTIFACTS_DIR / "columns.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("clear_data: saved wine_clean.csv rows=%d", df.shape[0])
    return True


def train_wrapper():
    """
    Импортируем train внутри функции, чтобы при парсинге DAG
    не выполнялся код из train_model.py (защита от побочных эффектов).
    """
    from train_model import train  # импорт локально
    return train()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# --- ВАЖНОЕ ИСПРАВЛЕНИЕ ---
# Убедитесь, что в конструктор DAG не передаётся unsupported kwargs (например, 'concurrency').
# Правильные параметры: dag_id, default_args, start_date, schedule_interval, catchup, max_active_runs и т.д.
dag = DAG(
    dag_id="wine_train_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
)

t_download = PythonOperator(
    task_id="download_data",
    python_callable=download_data,
    dag=dag,
)

t_clear = PythonOperator(
    task_id="clear_data",
    python_callable=clear_data,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_model",
    python_callable=train_wrapper,
    dag=dag,
)

t_download >> t_clear >> t_train
