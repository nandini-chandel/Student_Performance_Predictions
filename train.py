from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .utils import load_data, split_xy, train_val_split

def train_model(data_path: str, n_estimators: int = 200, max_depth: int | None = None, random_state: int = 42):
    df = load_data(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_val_split(X, y)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    importances = getattr(model, "feature_importances_", None)
    return model, metrics, importances, X.columns.tolist()
