import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.timemmd_dataset import load_domain_csv, get_feature_columns, TimeMMDWindowDataset
from src.models.factory import get_model_class  # 경로 수정
from src.utils.scaler import StandardScalerNP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="DLinear")  # 추가
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    df = load_domain_csv(args.csv_path)
    feature_cols = ckpt["feature_cols"]

    scaler = StandardScalerNP()
    scaler.mean = ckpt["scaler_mean"]
    scaler.std = ckpt["scaler_std"]

    ds = TimeMMDWindowDataset(
        df=df,
        feature_cols=feature_cols,
        target_col="OT",
        seq_len=ckpt["seq_len"],
        pred_len=ckpt["pred_len"],
        scaler=scaler,
        fit_scaler=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    ModelClass = get_model_class(args.model)
    args.enc_in = len(feature_cols)  # 추가
    model = ModelClass(args).to(args.device)  # device 수정
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    target_idx = feature_cols.index("OT")
    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(args.device)
            pred = model(x).cpu().numpy()
            pred_real = scaler.inverse_transform_feature(pred, target_idx)

            metas = batch["meta"]
            batch_size = pred_real.shape[0]
            for i in range(batch_size):
                row = {
                    "input_start_date": str(metas["input_start_date"][i]),
                    "input_end_date": str(metas["input_end_date"][i]),
                    "target_start_date": str(metas["target_start_date"][i]),
                    "target_end_date": str(metas["target_end_date"][i]),
                }
                for j in range(pred_real.shape[1]):
                    row[f"pred_t{j+1}"] = float(pred_real[i, j])
                rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv}")

if __name__ == "__main__":
    main()