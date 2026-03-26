import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.domains import DOMAIN_CONFIG
from src.datasets.timemmd_dataset import (
    load_domain_csv, time_split_df, get_feature_columns, TimeMMDWindowDataset
)
from models.factory import get_model_class
from src.utils.metrics import mae, mse, rmse, mape

def train_one_domain(args):
    domain_cfg = DOMAIN_CONFIG[args.domain]
    csv_path = Path(args.data_root) / domain_cfg["csv"]

    df = load_domain_csv(csv_path)
    split = time_split_df(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    feature_cols = get_feature_columns(df)

    train_ds = TimeMMDWindowDataset(
        split.train_df, feature_cols, "OT",
        seq_len=args.seq_len, pred_len=args.pred_len,
        scaler=None, fit_scaler=True
    )
    scaler = train_ds.scaler

    val_ds = TimeMMDWindowDataset(
        split.val_df, feature_cols, "OT",
        seq_len=args.seq_len, pred_len=args.pred_len,
        scaler=scaler, fit_scaler=False
    )
    test_ds = TimeMMDWindowDataset(
        split.test_df, feature_cols, "OT",
        seq_len=args.seq_len, pred_len=args.pred_len,
        scaler=scaler, fit_scaler=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)


    ModelClass = get_model_class(args.model)
    model = ModelClass(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(args.device)
                y = batch["y"].to(args.device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"[{args.domain}] Epoch {epoch+1:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt_dir = Path(args.output_root) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{args.domain}_pred{args.pred_len}.pt"
    torch.save({
        "model_state_dict": best_state,
        "feature_cols": feature_cols,
        "target_col": "OT",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
    }, ckpt_path)

    # test
    model.load_state_dict(best_state)
    model.to(args.device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(args.device)
            y = batch["y"].numpy()

            pred = model(x).cpu().numpy()
            preds.append(pred)
            trues.append(y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    target_idx = feature_cols.index("OT")
    preds_real = scaler.inverse_transform_feature(preds, target_idx)
    trues_real = scaler.inverse_transform_feature(trues, target_idx)

    metrics = {
        "domain": args.domain,
        "pred_len": args.pred_len,
        "mae": mae(preds_real, trues_real),
        "mse": mse(preds_real, trues_real),
        "rmse": rmse(preds_real, trues_real),
        "mape": mape(preds_real, trues_real),
    }

    metrics_dir = Path(args.output_root) / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / f"{args.domain}_pred{args.pred_len}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_one_domain(args)

if __name__ == "__main__":
    main()