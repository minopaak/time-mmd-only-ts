import subprocess
import sys
from configs.domains import DOMAIN_CONFIG

DATA_ROOT = "./data/Time-MMD/numerical"
OUTPUT_ROOT = "./outputs"

for domain, cfg in DOMAIN_CONFIG.items():
    seq_len = cfg["seq_len"]
    for pred_len in cfg["pred_lens"]:
        cmd = [
            sys.executable, "src/train.py",
            "--domain", domain,
            "--data_root", DATA_ROOT,
            "--output_root", OUTPUT_ROOT,
            "--model", "DLinear",  # 추가
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--epochs", "50",
            "--batch_size", "32",
            "--lr", "1e-3",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)