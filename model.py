import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

class XGB:
    def __init__(self, train_start: str, train_end: str, test_start: str, test_end: str, filename: str):
        self.df = pd.read_csv(f"./data/{filename}")
        if 'timestamp' not in self.df.columns:
            start_time = datetime.datetime.strptime("2025-01-01 10:00:00", "%Y-%m-%d %H:%M:%S")
            num_rows = len(self.df)
            time_deltas = [datetime.timedelta(seconds=i) for i in range(num_rows)]
            self.df['timestamp'] = [start_time + delta for delta in time_deltas]
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df_train = self.df[(self.df['timestamp'] >= train_start) & (self.df['timestamp'] <= train_end)]
        self.df_test = self.df[(self.df['timestamp'] >= test_start) & (self.df['timestamp'] <= test_end)]
        self.df_train.drop("timestamp", inplace=True, axis=1)
        self.df_test.drop("timestamp", inplace=True, axis=1)
        self.features_train = self.df_train.drop("Response", axis=1)
        self.target_train = self.df_train["Response"]
        self.features_test = self.df_test.drop("Response", axis=1)
        self.target_test = self.df_test["Response"]
        self.ypred = np.array([])
    
    def train_and_predict(self) -> np.array:
        neg_count = sum(self.target_train == 0)
        pos_count = sum(self.target_train == 1)
        scale_pos_weight_value = neg_count / pos_count
        print(scale_pos_weight_value)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight_value,
            eval_metric='logloss'
        )
        model.fit(self.features_train, self.target_train)
        ypred = model.predict(self.features_test)
        self.ypred = ypred
        return self.ypred
    
    def get_metrics(self):
        if len(self.ypred) != 0:
            accuracy = accuracy_score(self.target_test, self.ypred)
            precision = precision_score(self.target_test, self.ypred)
            recall = recall_score(self.target_test, self.ypred)
            f1 = f1_score(self.target_test, self.ypred)
            return {
                "error": None,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        else:
            return {
                "error": "The model hasn't been trained yet, therefore metrics can't be calculated",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }