from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

app = FastAPI()

class TrainRequest(BaseModel):
    dataset_id: str
    storage_root: str
    file_name: str = "processed.csv"
    target: str = "Response"
    model_name: str = "xgboost"
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # dataset_path: str | None = None  # uncomment if you want to accept it explicitly

@app.post("/train-model")
def train_model(req: TrainRequest):
    try:
        csv_path = Path(req.storage_root) / req.dataset_id / req.file_name
        # If you want to allow dataset_path override:
        # if req.dataset_path: csv_path = Path(req.dataset_path)

        if not csv_path.exists():
            raise HTTPException(status_code=400, detail=f"CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        if req.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target '{req.target}' not in CSV")

        ts_col = "synthetic_timestamp"
        if ts_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"'{ts_col}' not in CSV")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        t0, t1 = pd.to_datetime(req.train_start, utc=True), pd.to_datetime(req.train_end, utc=True)
        s0, s1 = pd.to_datetime(req.test_start,  utc=True), pd.to_datetime(req.test_end,  utc=True)

        train_df = df[(df[ts_col] >= t0) & (df[ts_col] <= t1)]
        test_df  = df[(df[ts_col] >= s0) & (df[ts_col] <= s1)]
        if train_df.empty: raise HTTPException(400, "Training slice is empty")
        if test_df.empty:  raise HTTPException(400, "Test slice is empty")

        # Minimal baseline so you get metrics even if feature engineering isn’t done yet
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.ensemble import RandomForestClassifier

        y_train = train_df[req.target].astype(int)
        y_test  = test_df[req.target].astype(int)
        X_train = train_df.select_dtypes(include=["number"]).drop(columns=[req.target], errors="ignore")
        X_test  = test_df.select_dtypes(include=["number"]).drop(columns=[req.target], errors="ignore")
        if X_train.empty: raise HTTPException(400, "No numeric features after filtering")

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        pre = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1  = float(f1_score(y_test, preds, zero_division=0))

        tp = int(((preds == 1) & (y_test.values == 1)).sum())
        tn = int(((preds == 0) & (y_test.values == 0)).sum())
        fp = int(((preds == 1) & (y_test.values == 0)).sum())
        fn = int(((preds == 0) & (y_test.values == 1)).sum())

        return {
            "modelId": "rf-baseline",
            "metrics": { "accuracy": acc*100, "precision": pre*100, "recall": rec*100, "f1": f1*100 },
            "confusion": { "tp": tp, "tn": tn, "fp": fp, "fn": fn }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
