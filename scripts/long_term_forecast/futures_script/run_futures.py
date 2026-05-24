"""
期货日频预测 - 精炼特征版（单模型，3组参数）
用法: python scripts/long_term_forecast/futures_script/run_futures.py
"""

import subprocess
import sys
import os

# 切到项目根目录
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
ROOT = os.path.abspath(ROOT)
os.chdir(ROOT)

MODEL = "iTransformer"
# MODEL = "DLinear"
SYMBOL = "RB"

EXPERIMENTS = [
    {"seq": 10, "pred": 1, "desc": "pred next 1 day"},
    # {"seq": 10, "pred": 3, "desc": "pred next 3 days"},
    # {"seq": 20, "pred": 1, "desc": "pred next 1 day with longer lookback"},
]

COMMON_ARGS = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/futures/",
    "--model", MODEL,
    "--data", "custom",
    "--features", "MS",
    "--target", "log_return",
    "--freq", "b",
    "--label_len", "5",
    "--e_layers", "2",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "5",
    "--dec_in", "5",
    "--c_out", "5",
    "--d_model", "64",
    "--d_ff", "128",
    "--des", "Exp",
    "--itr", "1",
    "--batch_size", "32",
    "--learning_rate", "0.003",
    "--train_epochs", "50",
    "--patience", "10",
    "--lradj", "cosine",
    "--num_workers", "0",
    "--loss_type", "classification",
]

print(f"{'='*60}")
print(f"  Model: {MODEL}   Symbol: {SYMBOL}")
print(f"{'='*60}")

for i, exp in enumerate(EXPERIMENTS, 1):
    seq, pred = exp["seq"], exp["pred"]
    print(f"\n[{i}/{len(EXPERIMENTS)}] seq={seq}, pred={pred} --- {exp['desc']}")
    print("-" * 60)

    cmd = [
        sys.executable, "-u", "run.py",
        "--data_path", f"{SYMBOL}.csv",
        "--model_id", f"{SYMBOL}_{seq}_{pred}",
        "--seq_len", str(seq),
        "--pred_len", str(pred),
    ] + COMMON_ARGS

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print(f"  All experiments done!")
print(f"{'='*60}")
