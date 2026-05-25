"""
RL 期货交易测试/评估脚本
用法:
    python test_rl.py --symbol RB --model_path ./rl_checkpoints/RB/best/best_model.zip
"""

import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from rl.config import RLConfig
from rl.env.futures_env import FuturesEnvTest
from rl.feature_extractor import iTransformerFeatureExtractor


def run_episode(model, env, deterministic=True):
    """运行一个完整 episode"""
    obs, info = env.reset()
    done = False

    records = []
    net_worths = []
    positions = []
    prices = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        records.append({
            "step": info["step"],
            "position": info["position"],
            "net_worth": info["net_worth"],
            "reward": info["reward"],
            "total_reward": info["total_reward"],
            "price": info.get("price", 0),
        })
        net_worths.append(info["net_worth"])
        positions.append(info["position"])
        prices.append(info.get("price", 0))

    return records, np.array(net_worths), np.array(positions), np.array(prices)


def compute_metrics(net_worths: np.ndarray, initial_capital: float):
    """计算绩效指标"""
    returns = np.diff(net_worths) / net_worths[:-1]
    total_return = (net_worths[-1] - initial_capital) / initial_capital

    # 年化 (假设 ~245 交易日, ~270 分钟/日)
    n_bars = len(net_worths)
    trading_days = n_bars / 270
    ann_return = (1 + total_return) ** (245 / max(trading_days, 1)) - 1 if total_return > -1 else -1

    # 波动率
    ann_vol = np.std(returns) * np.sqrt(245 * 270) if len(returns) > 1 else 0

    # Sharpe (无风险利率 2%)
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(net_worths)
    drawdown = (peak - net_worths) / peak
    max_dd = np.max(drawdown)

    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_bars": n_bars,
        "trading_days": trading_days,
        "final_net_worth": net_worths[-1],
    }


def print_report(metrics: dict, symbol: str):
    """打印绩效报告"""
    print(f"\n{'='*50}")
    print(f"  RL 策略评估报告 — {symbol}")
    print(f"{'='*50}")
    print(f"  总交易分钟数: {metrics['total_bars']}")
    print(f"  交易天数:     {metrics['trading_days']:.1f}")
    print(f"  最终净值:     {metrics['final_net_worth']:,.0f}")
    print(f"  总收益率:     {metrics['total_return']:.2%}")
    print(f"  年化收益率:   {metrics['annualized_return']:.2%}")
    print(f"  年化波动率:   {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe比率:   {metrics['sharpe_ratio']:.4f}")
    print(f"  最大回撤:     {metrics['max_drawdown']:.2%}")
    print(f"  Calmar比率:   {metrics['calmar_ratio']:.4f}")
    print(f"{'='*50}")


def plot_equity(net_worths, positions, prices, symbol, save_path=None):
    """绘制净值曲线和持仓"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 尝试设置中文字体
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 2]})

    # Net worth curve
    axes[0].plot(net_worths, linewidth=1, label="Net Worth")
    axes[0].axhline(y=net_worths[0], color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Net Worth")
    axes[0].set_title(f"RL Strategy - {symbol}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Position
    pos_colors = {1: "red", -1: "green", 0: "gray"}
    pos_labels = {1: "Long", -1: "Short", 0: "Flat"}
    for pos_val, color in pos_colors.items():
        mask = positions == pos_val
        axes[1].scatter(np.where(mask)[0], positions[mask], c=color, s=2,
                        label=pos_labels[pos_val])
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Short", "Flat", "Long"])
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Price
    axes[2].plot(prices, linewidth=0.8, color="steelblue")
    axes[2].set_ylabel("Price")
    axes[2].set_xlabel("Bar")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RL 期货交易评估")
    parser.add_argument("--symbol", type=str, default="RB")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径 (.zip)")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--features_dim", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    config = RLConfig(
        symbol=args.symbol,
        lookback_window=args.lookback,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        features_extractor_dim=args.features_dim,
        initial_capital=args.capital,
    )

    env = FuturesEnvTest(config)

    # 加载模型
    custom_objects = {
        "features_extractor_class": iTransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_extractor_dim,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "e_layers": config.e_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
        },
    }
    model = PPO.load(args.model_path, env=env, custom_objects=custom_objects)

    # 运行多个 episode
    all_metrics = []
    for ep in range(args.n_episodes):
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} ---")
        records, net_worths, positions, prices = run_episode(
            model, env, deterministic=args.deterministic
        )
        metrics = compute_metrics(net_worths, config.initial_capital)
        print_report(metrics, config.symbol)
        all_metrics.append(metrics)

        # 绘图 (最后一个 episode)
        if ep == args.n_episodes - 1:
            save_dir = os.path.join("./rl_results", config.symbol)
            os.makedirs(save_dir, exist_ok=True)
            plot_equity(
                net_worths, positions, prices, config.symbol,
                save_path=os.path.join(save_dir, f"rl_equity_{config.symbol}.png"),
            )

    # 汇总
    if args.n_episodes > 1:
        avg_sharpe = np.mean([m["sharpe_ratio"] for m in all_metrics])
        avg_return = np.mean([m["annualized_return"] for m in all_metrics])
        avg_dd = np.mean([m["max_drawdown"] for m in all_metrics])
        print(f"\n=== {args.n_episodes} 次 Episode 平均 ===")
        print(f"  平均年化收益: {avg_return:.2%}")
        print(f"  平均 Sharpe:  {avg_sharpe:.4f}")
        print(f"  平均最大回撤: {avg_dd:.2%}")


if __name__ == "__main__":
    main()
