# main.py
# FastAPI ML microservice for IntelliInspect - Training endpoint

from __future__ import annotations
from datetime import datetime

import os
from datetime import datetime
from typing import Optional, Literal, Tuple
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------
# Pydantic models (snake_case to match your .NET TrainingService payload)
# ---------------------------------------------------------------------
def to_naive_utc(ts: datetime) -> pd.Timestamp:
    """Convert any datetime to tz-naive UTC pd.Timestamp."""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    else:
        t = t.tz_localize(None)
    return t

def normalize_timestamp_series(s: pd.Series) -> pd.Series:
    """Coerce series to tz-naive UTC (consistent with to_naive_utc)."""
    s2 = pd.to_datetime(s, errors="coerce", utc=True)
    return s2.dt.tz_localize(None)

class TrainRequest(BaseModel):
    # 👇 this line silences the warning
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    storage_root: str
    dataset_path: str | None = None
    file_name: str | None = "processed.csv"
    target: str = "Response"
    model_name: str = "rf"          # keep your existing field name
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class TrainResponse(BaseModel):
    modelId: str
    metrics: dict
    confusion: dict
    charts: dict
    diagnostics: dict | None = None
    statusMessage: str


# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(title="IntelliInspect ML Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Helpers
# -----------------------

TIMESTAMP_COL = "synthetic_timestamp"


def resolve_csv_path(req: TrainRequest) -> str:
    if req.dataset_path:
        return req.dataset_path
    base = os.path.abspath(req.storage_root)
    path = os.path.join(base, req.dataset_id, req.file_name or "processed.csv")
    return path


def load_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    cols = _peek_columns(csv_path)
    parse_dates = [TIMESTAMP_COL] if TIMESTAMP_COL in cols else None

    df = pd.read_csv(csv_path, low_memory=False, parse_dates=parse_dates)

    # Force timestamp column to tz-naive UTC for consistent comparisons
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = normalize_timestamp_series(df[TIMESTAMP_COL])

    return df



def _peek_columns(csv_path: str, n: int = 1) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=n).columns)


def slice_by_time(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Required column '{TIMESTAMP_COL}' not found.")

    start_naive = to_naive_utc(start)
    end_naive   = to_naive_utc(end)

    mask = (df[TIMESTAMP_COL] >= start_naive) & (df[TIMESTAMP_COL] <= end_naive)
    return df.loc[mask].copy()



def build_pipeline(model_name: str, class_weight_scale: float) -> Pipeline:
    """
    A simple, robust preprocessing + model pipeline:
      - Numeric: median impute
      - Categorical: most_frequent impute + ordinal encode via factorize (inside a FunctionTransformer)
      - Model: RandomForest (default) using class_weight-balanced
    """
    from sklearn.preprocessing import FunctionTransformer

    def split_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        num = X.select_dtypes(include=[np.number])
        cat = X.select_dtypes(exclude=[np.number])
        return num, cat

    def encode_cat(cat_df: pd.DataFrame) -> np.ndarray:
        if cat_df.shape[1] == 0:
            return np.empty((len(cat_df), 0))
        # Cheap ordinal encoding via factorize per column
        encoded = []
        for c in cat_df.columns:
            codes, _ = pd.factorize(cat_df[c].astype("string"), sort=True)
            encoded.append(codes)
        return np.vstack(encoded).T

    numeric_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    pre_numeric = Pipeline(steps=[("imputer", numeric_imputer)])
    pre_categorical = Pipeline(
        steps=[
            ("imputer", cat_imputer),
            ("to_array", FunctionTransformer(lambda X: encode_cat(pd.DataFrame(X, columns=getattr(X, "columns", None))), validate=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", pre_numeric, lambda X: X.select_dtypes(include=[np.number]).columns),
            ("cat", pre_categorical, lambda X: X.select_dtypes(exclude=[np.number]).columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model selection (default to RF; xgboost optional if installed)
    if model_name in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                scale_pos_weight=class_weight_scale,
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=42,
            )
        except Exception:
            # Fallback to RF if xgboost not available
            model = RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
    else:
        # "rf" or "rf-baseline"
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, train_pos_rate: float) -> float:
    """
    Pick a threshold that maximizes F1; if degenerate, fall back to a quantile
    based on expected positive rate.
    """
    if proba.ndim != 1:
        proba = proba.ravel()

    prec, rec, th = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.nanargmax(f1s) if len(f1s) else 0
    threshold = float(th[max(0, min(idx, len(th) - 1))]) if len(th) else 0.5

    if not np.isfinite(threshold):
        expected_rate = max(0.005, float(train_pos_rate))
        threshold = float(np.quantile(proba, 1 - expected_rate))
    return threshold


# -----------------------
# Training endpoint
# -----------------------

@app.post("/train-model", response_model=TrainResponse)
def train_model(req: TrainRequest):
    csv_path = resolve_csv_path(req)

    try:
        df = load_dataframe(csv_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load dataset: {e}")

    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target}' not found in CSV.")

    # Split train/test by synthetic time
    try:
        train_df = slice_by_time(df, req.train_start, req.train_end)
        test_df = slice_by_time(df, req.test_start, req.test_end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Time slicing error: {e}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise HTTPException(status_code=400, detail="Train or test slice is empty. Widen your windows.")

    # Separate features/target and drop timestamp
    drop_cols = [req.target]
    if TIMESTAMP_COL in train_df.columns:
        drop_cols.append(TIMESTAMP_COL)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[req.target].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[req.target].astype(int)

    # Basic sanity: if y has only one class in train, the model may collapse
    pos_train = int((y_train == 1).sum())
    neg_train = int((y_train == 0).sum())
    train_pos_rate = (pos_train / max(1, (pos_train + neg_train))) if (pos_train + neg_train) > 0 else 0.0

    # Build pipeline & fit
    scale = (neg_train / max(1, pos_train)) if pos_train > 0 else 1.0
    pipe = build_pipeline(req.model_name or "rf", class_weight_scale=scale)
    pipe.fit(X_train, y_train)

    # Predict probabilities on test
    # (Pipeline ensures we have predict_proba)
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model cannot produce probabilities: {e}")

    # Choose a threshold that gives some positives when appropriate
    threshold = choose_threshold(y_test.values, proba, train_pos_rate)
    preds = (proba >= threshold).astype(int)

    # Metrics
    tp = int(((preds == 1) & (y_test.values == 1)).sum())
    tn = int(((preds == 0) & (y_test.values == 0)).sum())
    fp = int(((preds == 1) & (y_test.values == 0)).sum())
    fn = int(((preds == 0) & (y_test.values == 1)).sum())

    acc = float(accuracy_score(y_test, preds))
    pre = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1  = float(f1_score(y_test, preds, zero_division=0))

    # Build response (percentages for UI)
    resp = TrainResponse(
        modelId=(req.model_name or "rf") + "-baseline",
        metrics={
            "accuracy": acc * 100.0,
            "precision": pre * 100.0,
            "recall": rec * 100.0,
            "f1": f1 * 100.0,
        },
        confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        charts={
            # Fill later with base64 if you choose to render charts
            "trainingAccuracyBase64": None,
            "trainingLossBase64": None,
            "confusionMatrixBase64": None,
        },
        diagnostics={
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_pos": pos_train,
            "train_neg": neg_train,
            "threshold": threshold,
            "proba_min": float(np.min(proba)),
            "proba_max": float(np.max(proba)),
            "proba_mean": float(np.mean(proba)),
        },
        statusMessage="Model Trained Successfully",
    )
    return resp


# -----------------------
# Run (for local dev)
# -----------------------
if __name__ == "__main__":
    # Run with: python -m uvicorn main:app --reload --port 8000
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
