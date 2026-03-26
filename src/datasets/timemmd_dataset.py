from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.scaler import StandardScalerNP

DATE_COLS = ["start_date", "end_date"]
TARGET_COL = "OT"

@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

def load_domain_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"])
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values("start_date").reset_index(drop=True)
    return df

def time_split_df(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.1) -> SplitData:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return SplitData(
        train_df=df.iloc[:train_end].reset_index(drop=True),
        val_df=df.iloc[train_end:val_end].reset_index(drop=True),
        test_df=df.iloc[val_end:].reset_index(drop=True),
    )

def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in DATE_COLS]

class TimeMMDWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        pred_len: int,
        scaler: StandardScalerNP | None = None,
        fit_scaler: bool = False,
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.target_idx = feature_cols.index(target_col)
        self.seq_len = seq_len
        self.pred_len = pred_len

        values = self.df[self.feature_cols].values.astype(np.float32)

        self.scaler = scaler if scaler is not None else StandardScalerNP()
        if fit_scaler:
            self.scaler.fit(values)

        self.values = self.scaler.transform(values).astype(np.float32)
        self.start_dates = self.df["start_date"].values if "start_date" in self.df.columns else None
        self.end_dates = self.df["end_date"].values if "end_date" in self.df.columns else None

    def __len__(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.values[idx : idx + self.seq_len]  # [seq_len, num_features]
        y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]  # [pred_len]

        meta = {
            "input_start_date": self.start_dates[idx] if self.start_dates is not None else None,
            "input_end_date": self.end_dates[idx + self.seq_len - 1] if self.end_dates is not None else None,
            "target_start_date": self.start_dates[idx + self.seq_len] if self.start_dates is not None else None,
            "target_end_date": self.end_dates[idx + self.seq_len + self.pred_len - 1] if self.end_dates is not None else None,
        }

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "meta": meta,
        }