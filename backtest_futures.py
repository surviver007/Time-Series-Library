"""
期货预测回测系统
加载训练好的模型预测结果 -> 概率估计 -> 信号生成 -> 交易模拟 -> 绩效报告

用法:
  python backtest_futures.py --model DLinear --symbol RB
  python backtest_futures.py --compare --symbol RB
  python backtest_futures.py --compare --symbol RB --threshold 0.55
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. 加载预测结果 & 日期对齐
# ============================================================

def find_setting_dir(symbol, seq, pred, model):
    """根据 symbol/seq/pred/model 找到 results 下的目录"""
    # 匹配 run_futures_compare.py 生成的目录名
    # 格式: long_term_forecast_{symbol}_{seq}_{pred}_{model}_*...
    pattern = os.path.join("results", f"long_term_forecast_{symbol}_{seq}_{pred}_{model}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到: {pattern}")
    # 可能有不同超参的多次运行，取最新的
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def load_predictions(results_dir):
    """加载 pred.npy 和 true.npy"""
    pred_path = os.path.join(results_dir, "pred.npy")
    true_path = os.path.join(results_dir, "true.npy")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"pred.npy 不存在: {pred_path}")
    pred = np.load(pred_path).squeeze()  # (N, pred_len, 1) -> (N,)
    true = np.load(true_path).squeeze()
    return pred, true


def load_dates_and_scaler(csv_path, seq_len, pred_len):
    """
    加载原始 CSV，复现 Dataset_Custom 的 train/val/test split，
    返回测试集日期、scaler 的 mean/std（用于反标准化）

    日期对齐逻辑:
      Dataset_Custom test border1 = N - num_test - seq_len
      test 样本 i 的预测目标是原始数据第 (N - num_test + i) 行
    """
    df = pd.read_csv(csv_path)
    n = len(df)

    num_train = int(n * 0.7)
    num_test = int(n * 0.2)

    # 测试样本数 = num_test - pred_len + 1
    num_test_samples = num_test - pred_len + 1
    dates = [df['date'].iloc[n - num_test + i] for i in range(num_test_samples)]

    # 复现 scaler: fit on train split
    feature_cols = [c for c in df.columns if c not in ['date']]
    train_data = df[feature_cols].iloc[:num_train].values.astype(np.float64)

    # sklearn StandardScaler 使用 ddof=0
    means = train_data.mean(axis=0)
    stds = train_data.std(axis=0, ddof=0)

    # 验证长度是否一致
    return dates, means, stds


def inverse_transform_target(pred, means, stds):
    """反标准化: 只对 target 列（最后一列 = log_return）"""
    return pred * stds[-1] + means[-1]


# ============================================================
# 2. 概率估计
# ============================================================

def compute_probability(pred_orig, sigma):
    """
    P(上涨) = P(true_log_return > 0 | pred)
    假设 true ~ Normal(pred, sigma^2)
    P(true > 0) = Phi(pred / sigma)
    """
    z = pred_orig / sigma
    return stats.norm.cdf(z)


# ============================================================
# 3. 信号生成
# ============================================================

def generate_signals(prob_up, threshold):
    """
    P(up) > threshold  -> +1 做多
    P(down) > threshold -> -1 做空
    否则 -> 0 空仓
    """
    signals = np.zeros(len(prob_up))
    signals[prob_up > threshold] = 1
    signals[prob_up < (1 - threshold)] = -1
    return signals.astype(int)


# ============================================================
# 4. 交易模拟
# ============================================================

def simulate_trading(signals, actual_returns, commission=0.0001, slippage=0.0001):
    """
    逐日模拟交易
    signals:        每日信号 (+1/-1/0)
    actual_returns: 每日实际 log_return
    commission:     单边手续费率
    slippage:       滑点率
    """
    n = len(signals)
    daily_returns = np.zeros(n)
    prev_pos = 0
    cost_per_unit = commission + slippage
    num_trades = 0

    for i in range(n):
        pos = signals[i]

        # 换仓成本
        if pos != prev_pos:
            daily_returns[i] -= abs(pos - prev_pos) * cost_per_unit
            num_trades += 1

        # 持仓收益
        daily_returns[i] += pos * actual_returns[i]
        prev_pos = pos

    return daily_returns, num_trades


# ============================================================
# 5. 绩效指标
# ============================================================

def compute_metrics(daily_returns, signals, dates):
    """计算完整绩效指标"""
    n = len(daily_returns)
    trading_days = 242  # 年交易日

    # 净值曲线
    equity = np.cumprod(1 + daily_returns)

    # 总收益率
    total_return = equity[-1] - 1

    # 年化收益率
    annual_return = (1 + total_return) ** (trading_days / n) - 1 if n > 0 else 0

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(trading_days) if n > 1 else 0

    # Sharpe Ratio (无风险利率 2%)
    rf_daily = 0.02 / trading_days
    excess = daily_returns - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(trading_days)) if excess.std() > 0 else 0

    # 最大回撤
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # 胜率 & 盈亏比（只看有持仓的日子）
    active = daily_returns[signals != 0]
    if len(active) > 0:
        win_rate = (active > 0).mean()
        wins = active[active > 0]
        losses = active[active < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-10
        profit_factor = avg_win / avg_loss
    else:
        win_rate = 0
        profit_factor = 0

    # 信号分布
    long_pct = (signals == 1).sum() / n * 100
    short_pct = (signals == -1).sum() / n * 100
    flat_pct = (signals == 0).sum() / n * 100

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(np.where(np.diff(signals, prepend=0) != 0)[0]),
        'long_pct': long_pct,
        'short_pct': short_pct,
        'flat_pct': flat_pct,
        'start_date': dates[0],
        'end_date': dates[-1],
        'num_days': n,
        'equity': equity,
    }


def print_report(m, model_name, threshold):
    """打印绩效报告"""
    print(f"\n{'='*60}")
    print(f"  {model_name}  |  threshold={threshold}")
    print(f"  {m['start_date']} ~ {m['end_date']}  ({m['num_days']} days)")
    print(f"{'='*60}")

    print(f"\n  Returns")
    print(f"    Total:       {m['total_return']:>+8.2%}")
    print(f"    Annualized:  {m['annual_return']:>+8.2%}")
    print(f"    Ann. Vol:    {m['annual_vol']:>8.2%}")

    print(f"\n  Risk")
    print(f"    Sharpe:      {m['sharpe']:>8.2f}")
    print(f"    Max DD:      {m['max_dd']:>8.2%}")
    print(f"    Calmar:      {m['calmar']:>8.2f}")

    print(f"\n  Trading")
    print(f"    Win Rate:    {m['win_rate']:>8.2%}")
    print(f"    Profit/LOSS: {m['profit_factor']:>8.2f}")
    print(f"    Trades:      {m['num_trades']:>8d}")

    print(f"\n  Signal Distribution")
    print(f"    Long:  {m['long_pct']:>5.1f}%")
    print(f"    Short: {m['short_pct']:>5.1f}%")
    print(f"    Flat:  {m['flat_pct']:>5.1f}%")


# ============================================================
# 6. 单模型回测
# ============================================================

def run_backtest(symbol, seq, pred, model, threshold, commission, slippage, loss_type='regression'):
    """完整回测流程"""
    results_dir = find_setting_dir(symbol, seq, pred, model)
    print(f"\n[Model: {model}]  results: {os.path.basename(results_dir)}")

    # 加载预测
    pred_scaled, true_scaled = load_predictions(results_dir)

    # 加载日期和 scaler
    csv_path = os.path.join("dataset", "futures", f"{symbol}.csv")
    dates, means, stds = load_dates_and_scaler(csv_path, seq, pred)

    # 对齐长度
    n = min(len(pred_scaled), len(dates))
    if len(pred_scaled) != len(dates):
        print(f"  WARNING: pred({len(pred_scaled)}) != dates({len(dates)}), using min={n}")
    pred_scaled = pred_scaled[:n]
    true_scaled = true_scaled[:n]
    dates = dates[:n]

    # 反标准化
    pred_orig = inverse_transform_target(pred_scaled, means, stds)
    true_orig = inverse_transform_target(true_scaled, means, stds)

    # 概率估计
    if loss_type == 'classification':
        # 分类模式: 模型输出就是 logit，直接 sigmoid 得到 P(上涨)
        prob_up = 1 / (1 + np.exp(-pred_orig))
        sigma = None
        print(f"  mode=classification, using sigmoid directly")
    else:
        # 回归模式: 正态分布假设
        residuals = true_orig - pred_orig
        sigma = residuals.std()
        print(f"  mode=regression, sigma(residual) = {sigma:.6f}")
        prob_up = compute_probability(pred_orig, sigma)

    # 信号 & 交易
    signals = generate_signals(prob_up, threshold)
    daily_returns, _ = simulate_trading(signals, true_orig, commission, slippage)
    metrics = compute_metrics(daily_returns, signals, dates)

    return metrics, signals, daily_returns, dates, pred_orig, true_orig, prob_up, sigma


# ============================================================
# 7. 多模型对比
# ============================================================

MODELS = ["DLinear", "TimesNet", "iTransformer", "PatchTST"]


def run_compare(symbol, seq, pred, threshold, commission, slippage, loss_type='regression'):
    """对比所有模型"""
    all_results = {}

    for model in MODELS:
        try:
            result = run_backtest(symbol, seq, pred, model, threshold, commission, slippage, loss_type)
            all_results[model] = result
        except FileNotFoundError as e:
            print(f"  Skip {model}: {e}")

    if not all_results:
        print("No results found!")
        return

    # 买入持有基准
    csv_path = os.path.join("dataset", "futures", f"{symbol}.csv")
    dates, means, stds = load_dates_and_scaler(csv_path, seq, pred)
    any_model = list(all_results.values())[0]
    n = len(any_model[4])  # pred_orig
    true_orig = any_model[5][:n]
    bh_returns = true_orig  # buy & hold = daily log_return
    bh_equity = np.cumprod(1 + bh_returns)
    bh_total = bh_equity[-1] - 1
    bh_annual = (1 + bh_total) ** (242 / n) - 1
    bh_vol = bh_returns.std() * np.sqrt(242)
    bh_sharpe = (bh_returns.mean() - 0.02/242) / bh_returns.std() * np.sqrt(242) if bh_returns.std() > 0 else 0
    bh_dd = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()

    # 对比表
    print(f"\n{'='*80}")
    print(f"  Model Comparison  |  {symbol}  |  threshold={threshold}")
    print(f"{'='*80}")
    header = f"  {'Model':<14} {'AnnRet':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'Long%':>6} {'Short%':>6}"
    print(header)
    print(f"  {'-'*76}")

    # 基准
    print(f"  {'Buy&Hold':<14} {bh_annual:>+8.2%} {bh_sharpe:>8.2f} {bh_dd:>7.2%} {'N/A':>8} {'N/A':>7} {'100%':>6} {'0%':>6}")

    for model, (m, *_) in sorted(all_results.items(), key=lambda x: -x[1][0]['sharpe']):
        print(f"  {model:<14} {m['annual_return']:>+8.2%} {m['sharpe']:>8.2f} {m['max_dd']:>7.2%} {m['win_rate']:>7.2%} {m['num_trades']:>7d} {m['long_pct']:>5.1f}% {m['short_pct']:>5.1f}%")

    # 各模型详细报告
    for model, (m, signals, daily_returns, dates, pred_orig, true_orig, prob_up, sigma) in all_results.items():
        print_report(m, model, threshold)

    # 绘制对比图
    plot_compare(all_results, symbol, threshold, dates, bh_equity, true_orig)

    return all_results


# ============================================================
# 8. 可视化
# ============================================================

def plot_results(model_name, metrics, dates, pred_orig, true_orig, prob_up, signals, threshold):
    """单模型回测可视化"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 不可用，跳过绘图")
        return

    equity = metrics['equity']
    n = len(equity)
    benchmark = np.cumprod(1 + true_orig)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{model_name} | Sharpe={metrics["sharpe"]:.2f} | MaxDD={metrics["max_dd"]:.2%}', fontsize=14)

    x = range(n)

    # 1. 净值曲线
    axes[0].plot(x, equity, label=f'{model_name}', linewidth=1.5)
    axes[0].plot(x, benchmark, label='Buy & Hold', linewidth=1, alpha=0.6, color='gray')
    axes[0].set_ylabel('Equity')
    axes[0].legend()
    axes[0].set_title('Equity Curve')
    axes[0].grid(True, alpha=0.3)

    # 2. 回撤
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    axes[1].fill_between(x, dd, 0, color='red', alpha=0.3)
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title('Drawdown')
    axes[1].grid(True, alpha=0.3)

    # 3. 概率 & 信号
    axes[2].plot(x, prob_up, linewidth=0.8, label='P(up)', color='steelblue')
    axes[2].axhline(y=threshold, color='green', linestyle='--', alpha=0.5, label=f'+{threshold}')
    axes[2].axhline(y=1-threshold, color='red', linestyle='--', alpha=0.5, label=f'-{threshold}')
    axes[2].axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    long_idx = np.where(signals == 1)[0]
    short_idx = np.where(signals == -1)[0]
    axes[2].scatter(long_idx, prob_up[long_idx], c='green', s=8, alpha=0.4, label='Long')
    axes[2].scatter(short_idx, prob_up[short_idx], c='red', s=8, alpha=0.4, label='Short')
    axes[2].set_ylabel('P(up)')
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc='upper right', fontsize=7)
    axes[2].set_title('Prediction Probability & Signals')
    axes[2].grid(True, alpha=0.3)

    # 4. 实际收益
    colors = ['green' if r > 0 else 'red' for r in true_orig]
    axes[3].bar(x, true_orig, color=colors, alpha=0.5, width=1)
    axes[3].set_ylabel('log_return')
    axes[3].set_title('Actual Daily Returns')
    axes[3].grid(True, alpha=0.3)

    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels([dates[i] for i in ticks], rotation=45, fontsize=8)

    plt.tight_layout()
    out_path = f"backtest_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved: {out_path}")
    plt.close()


def plot_compare(all_results, symbol, threshold, dates, bh_equity, true_orig):
    """多模型净值对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(true_orig)
    x = range(n)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, bh_equity, label='Buy & Hold', linewidth=1.5, color='gray', alpha=0.7)

    colors = {'DLinear': 'blue', 'TimesNet': 'orange', 'iTransformer': 'green', 'PatchTST': 'purple'}
    for model, (m, *_) in all_results.items():
        ax.plot(x, m['equity'][:n], label=f"{model} (Sharpe={m['sharpe']:.2f})", linewidth=1.2, color=colors.get(model, 'black'))

    ax.set_title(f'{symbol} Multi-Model Comparison | threshold={threshold}')
    ax.set_ylabel('Equity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks], rotation=45, fontsize=8)

    plt.tight_layout()
    out_path = f"backtest_compare_{symbol}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Comparison chart saved: {out_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Futures Prediction Backtest')
    parser.add_argument('--model', type=str, default='DLinear',
                        help='Model name (DLinear/TimesNet/iTransformer/PatchTST)')
    parser.add_argument('--symbol', type=str, default='RB')
    parser.add_argument('--seq', type=int, default=10)
    parser.add_argument('--pred', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Probability threshold for signals')
    parser.add_argument('--commission', type=float, default=0.0001,
                        help='Single-side commission rate')
    parser.add_argument('--slippage', type=float, default=0.0001,
                        help='Slippage rate')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['regression', 'classification'],
                        help='loss type: regression (normal assumption) or classification (sigmoid)')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

    if args.compare:
        run_compare(args.symbol, args.seq, args.pred, args.threshold,
                    args.commission, args.slippage, args.loss_type)
    else:
        result = run_backtest(args.symbol, args.seq, args.pred, args.model,
                              args.threshold, args.commission, args.slippage, args.loss_type)
        metrics, signals, daily_returns, dates, pred_orig, true_orig, prob_up, sigma = result
        print_report(metrics, args.model, args.threshold)
        plot_results(args.model, metrics, dates, pred_orig, true_orig, prob_up, signals, args.threshold)


if __name__ == '__main__':
    main()
