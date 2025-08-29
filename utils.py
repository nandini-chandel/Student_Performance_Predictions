from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Hours_Studied", "Attendance", "Past_Score", "Sleep_Hours", "Social_Media_Hours"]
TARGET = "Final_Score"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity cleanup
    df = df.dropna().copy()
    return df

def split_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
