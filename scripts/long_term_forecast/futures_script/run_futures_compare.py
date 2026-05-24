"""
期货日频预测 - 多模型对比（精炼特征）
统一参数 seq=10, pred=1，对比 4 个模型
用法: python scripts/long_term_forecast/futures_script/run_futures_compare.py
"""

import subprocess
import sys
import os

# 切到项目根目录
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
ROOT = os.path.abspath(ROOT)
os.chdir(ROOT)

SYMBOL = "RB"
SEQ = 10
PRED = 1

MODELS = [
    {"name": "DLinear",     "extra": []},
    {"name": "TimesNet",    "extra": ["--d_model", "32", "--d_ff", "64", "--top_k", "3"]},
    {"name": "iTransformer","extra": ["--d_model", "64", "--d_ff", "128"]},
    {"name": "PatchTST",    "extra": ["--d_model", "32", "--d_ff", "64", "--patch_len", "4"]},
]

COMMON_ARGS = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/futures/",
    "--data_path", f"{SYMBOL}.csv",
    "--data", "custom",
    "--features", "MS",
    "--target", "log_return",
    "--freq", "b",
    "--seq_len", str(SEQ),
    "--label_len", "5",
    "--pred_len", str(PRED),
    "--e_layers", "2",
    "--d_layers", "1",
    "--factor", "3",
    "--enc_in", "5",
    "--dec_in", "5",
    "--c_out", "5",
    "--des", "Exp",
    "--itr", "1",
    "--batch_size", "32",
    "--learning_rate", "0.0005",
    "--train_epochs", "50",
    "--patience", "5",
    "--lradj", "cosine",
    "--num_workers", "0",
    "--loss_type", "classification",
]

print(f"{'='*60}")
print(f"  Multi-Model Comparison   Symbol: {SYMBOL}   seq={SEQ}  pred={PRED}")
print(f"{'='*60}")

for i, m in enumerate(MODELS, 1):
    name = m["name"]
    print(f"\n[{i}/{len(MODELS)}] {name}")
    print("-" * 60)

    cmd = [
        sys.executable, "-u", "run.py",
        "--model_id", f"{SYMBOL}_{SEQ}_{PRED}_{name}",
        "--model", name,
    ] + COMMON_ARGS + m["extra"]

    subprocess.run(cmd, check=True)

# 汇总结果
print(f"\n{'='*60}")
print(f"  All models done! Results:")
print(f"{'='*60}")

result_file = os.path.join(ROOT, "result_long_term_forecast.txt")
if os.path.exists(result_file):
    with open(result_file, "r") as f:
        for line in f:
            if f"{SYMBOL}_{SEQ}_{PRED}" in line:
                print(f"  {line.strip()}")
else:
    print("  (result file not found)")
