"""
期货数据预处理：原始价格 -> 精炼特征
读取 dataset/futures_origin/ 下的原始 CSV，转换后保存到 dataset/futures/

改进点：
  1. 预测目标：从「价格」改为「对数收益率 log_return」
  2. 特征工程：从 8 维冗余价格字段 -> 5 维精炼特征
     - log_return:       对数收益率（核心预测目标）
     - high_low_range:   日内振幅 (high-low)/close，衡量波动
     - open_close_range: 实体幅度 (close-open)/close，衡量方向+力度
     - vol_change:       成交量变化率
     - oi_change:        持仓量变化率

用法: python preprocess_futures.py
"""

import os
import numpy as np
import pandas as pd

from download_futures import FUTURES

INPUT_DIR = "./dataset/futures_origin"
OUTPUT_DIR = "./dataset/futures"


def preprocess_one(filepath, output_path, symbol):
    """转换单个品种"""
    print(f"\n{'='*50}")
    print(f"处理 {symbol}: {filepath}")

    df = pd.read_csv(filepath)
    print(f"  原始数据: {len(df)} 条, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  原始列: {list(df.columns)}")

    # ============ 特征工程 ============

    # 1. 对数收益率（核心）
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. 日内振幅（波动性）
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # 3. 实体幅度（方向+力度）
    df['open_close_range'] = (df['close'] - df['open']) / df['close']

    # 4. 成交量变化率（用 log 变化，避免除零）
    df['vol_change'] = np.log(df['vol'] / df['vol'].shift(1))

    # 5. 持仓量变化率
    df['oi_change'] = np.log(df['oi'] / df['oi'].shift(1))

    # 去掉第一行（shift 产生 NaN）
    df = df.dropna().reset_index(drop=True)

    # 只保留精炼后的列
    # 列顺序：date, 特征列..., target（log_return 放最后作为预测目标）
    refined_cols = ['date', 'high_low_range', 'open_close_range', 'vol_change', 'oi_change', 'log_return']
    df_refined = df[refined_cols].copy()

    print(f"\n  精炼后列: {list(df_refined.columns)}")
    print(f"  精炼后数据: {len(df_refined)} 条")

    # ============ 统计信息 ============
    target = df_refined['log_return']
    print(f"\n  log_return 统计:")
    print(f"    均值:     {target.mean():.6f}")
    print(f"    标准差:   {target.std():.6f}")
    print(f"    最小值:   {target.min():.4f}")
    print(f"    最大值:   {target.max():.4f}")
    print(f"    偏度:     {target.skew():.4f}")
    print(f"    峰度:     {target.kurtosis():.4f}")

    # 方向分布
    up = (target > 0).sum()
    down = (target < 0).sum()
    flat = (target == 0).sum()
    print(f"    涨/跌/平: {up}/{down}/{flat}  ({up/len(target)*100:.1f}%/{down/len(target)*100:.1f}%/{flat/len(target)*100:.1f}%)")

    # 其他特征统计
    print(f"\n  其他特征统计:")
    for col in ['high_low_range', 'open_close_range', 'vol_change', 'oi_change']:
        s = df_refined[col]
        print(f"    {col:20s}: mean={s.mean():+.6f}, std={s.std():.6f}, range=[{s.min():.4f}, {s.max():.4f}]")

    # ============ 保存 ============
    df_refined.to_csv(output_path, index=False)
    print(f"\n 已保存: {output_path} ({len(df_refined)} 条)")

    return df_refined


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for symbol, info in FUTURES.items():
        input_path = os.path.join(INPUT_DIR, f"{symbol}.csv")
        output_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")

        if not os.path.exists(input_path):
            print(f" 跳过 {symbol}（{info['name']}）: 文件不存在 {input_path}")
            continue

        df = preprocess_one(input_path, output_path, f"{symbol}（{info['name']}）")
        results[symbol] = df

    # ============ 汇总 ============
    print(f"\n\n{'='*60}")
    print(f"全品种汇总")
    print(f"{'='*60}")
    print(f"{'品种':<8} {'条数':>6} {'均值':>10} {'标准差':>10} {'涨%':>6} {'跌%':>6}")
    print(f"{'-'*60}")
    for symbol, df in results.items():
        ret = df['log_return']
        up_pct = (ret > 0).sum() / len(ret) * 100
        down_pct = (ret < 0).sum() / len(ret) * 100
        print(f"{symbol:<8} {len(ret):>6} {ret.mean():>10.6f} {ret.std():>10.6f} {up_pct:>5.1f}% {down_pct:>5.1f}%")

    print(f"\n 全部完成！精炼数据保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
