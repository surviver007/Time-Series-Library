"""
检查模型预测结果：metrics.npy, pred.npy, true.npy
用法: python inspect_results.py <results_dir>
示例: python inspect_results.py results/long_term_forecast_xxx/
"""
import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("用法: python inspect_results.py <results_dir>")
    print("示例: python inspect_results.py results/long_term_forecast_futures_DLinear_futures_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/")
    sys.exit(1)

folder = sys.argv[1].rstrip('/')

# --- metrics.npy ---
metrics_path = os.path.join(folder, 'metrics.npy')
if os.path.exists(metrics_path):
    metrics = np.load(metrics_path)
    print("=" * 60)
    print("metrics.npy:")
    print(f"  shape: {metrics.shape}")
    print(f"  values: {metrics}")
    if len(metrics) == 2:
        print(f"  accuracy: {metrics[0]:.6f}")
        print(f"  auc:      {metrics[1]:.6f}")
    elif len(metrics) == 5:
        print(f"  mae:  {metrics[0]:.6f}")
        print(f"  mse:  {metrics[1]:.6f}")
        print(f"  rmse: {metrics[2]:.6f}")
        print(f"  mape: {metrics[3]:.6f}")
        print(f"  mspe: {metrics[4]:.6f}")
else:
    print(f"[跳过] {metrics_path} 不存在")

# --- pred.npy ---
pred_path = os.path.join(folder, 'pred.npy')
if os.path.exists(pred_path):
    pred = np.load(pred_path)
    print("\n" + "=" * 60)
    print("pred.npy:")
    print(f"  shape: {pred.shape}")
    print(f"  dtype: {pred.dtype}")
    print(f"  min:   {pred.min():.6f}")
    print(f"  max:   {pred.max():.6f}")
    print(f"  mean:  {pred.mean():.6f}")
    print(f"  std:   {pred.std():.6f}")
    print(f"  前5个值: {pred.flatten()[:5]}")
    # 分布直方图
    from scipy.special import expit as sigmoid
    prob = sigmoid(pred.flatten())
    print(f"\n  sigmoid后:")
    print(f"    min:  {prob.min():.6f}")
    print(f"    max:  {prob.max():.6f}")
    print(f"    mean: {prob.mean():.6f}")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(prob, bins=bins)
    print(f"    概率分布:")
    for i in range(len(bins) - 1):
        bar = '#' * (hist[i] // max(hist.max() // 50, 1))
        print(f"      [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:>8d}  {bar}")
else:
    print(f"\n[跳过] {pred_path} 不存在")

# --- true.npy ---
true_path = os.path.join(folder, 'true.npy')
if os.path.exists(true_path):
    true = np.load(true_path)
    print("\n" + "=" * 60)
    print("true.npy:")
    print(f"  shape: {true.shape}")
    print(f"  dtype: {true.dtype}")
    print(f"  min:   {true.min():.6f}")
    print(f"  max:   {true.max():.6f}")
    print(f"  mean:  {true.mean():.6f}")
    print(f"  前5个值: {true.flatten()[:5]}")

    unique, counts = np.unique(true.flatten(), return_counts=True)
    print(f"\n  唯一值分布:")
    for u, c in zip(unique, counts):
        total = counts.sum()
        pct = c / total * 100
        bar = '#' * int(pct)
        print(f"    {u:>8.4f}: {c:>8d}  ({pct:5.1f}%)  {bar}")
else:
    print(f"\n[跳过] {true_path} 不存在")

# --- 对比样例 ---
if os.path.exists(pred_path) and os.path.exists(true_path):
    print("\n" + "=" * 60)
    print("前20个样本对比 (sigmoid后):")
    prob_flat = sigmoid(pred.flatten())
    true_flat = true.flatten()
    pred_labels = (prob_flat >= 0.5).astype(int)
    true_labels = true_flat.astype(int)
    print(f"  {'idx':>4s}  {'pred_raw':>10s}  {'prob':>8s}  {'pred_lbl':>8s}  {'true_lbl':>8s}  {'match':>5s}")
    print(f"  {'-'*50}")
    for i in range(min(20, len(prob_flat))):
        match = "OK" if pred_labels[i] == true_labels[i] else "X"
        print(f"  {i:>4d}  {pred.flatten()[i]:>10.4f}  {prob_flat[i]:>8.4f}  {pred_labels[i]:>8d}  {true_labels[i]:>8d}  {match:>5s}")
