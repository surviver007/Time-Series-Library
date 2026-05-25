"""
期货分钟级数据特征工程 — RL 训练专用
读取 dataset/futures_origin_minute/{SYMBOL}.csv
输出 dataset/futures_rl/{SYMBOL}.csv + {SYMBOL}_stats.json

特征 (22 维市场特征):
  - 价格相关 (6): normalized_open, high_pct, low_pct, minute_return, volume_ratio, oi_change
  - 多尺度 return (4): log return at [5, 15, 30, 60] min
  - 多尺度 volatility (4): rolling std at [15, 30, 60, 120] min
  - 多尺度 MA_gap (4): (close - MA) / MA at [15, 30, 60, 120] min
  - 时间特征 (4): session_progress, minute_of_session, hour_sin, hour_cos

用法: python preprocess_futures_rl.py [--symbol RB]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

FUTURES = {
    "RB": {"name": "螺纹钢", "session_start": 9, "session_end": 15, "night": True},
    "I":  {"name": "铁矿石", "session_start": 9, "session_end": 15, "night": True},
    "AU": {"name": "黄金",   "session_start": 9, "session_end": 15, "night": True},
    "CU": {"name": "铜",     "session_start": 9, "session_end": 15, "night": True},
    "M":  {"name": "豆粕",   "session_start": 9, "session_end": 15, "night": False},
}

INPUT_DIR = "./dataset/futures_origin_minute"
OUTPUT_DIR = "./dataset/futures_rl"

RETURN_WINDOWS = [5, 15, 30, 60]
VOL_WINDOWS = [15, 30, 60, 120]
MA_WINDOWS = [15, 30, 60, 120]
NUM_MARKET_FEATURES = 22  # 6 + 4*3 + 4


def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """计算全部 22 个市场特征"""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df = df.copy()

    # 确保按时间排序
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ---- 价格相关特征 (6) ----
    df["norm_open"] = open_ / close.shift(1) - 1.0       # 相对前收开盘
    df["high_pct"] = (high - close) / close               # 上影线比例
    df["low_pct"] = (low - close) / close                 # 下影线比例
    with np.errstate(divide="ignore", invalid="ignore"):
        df["minute_return"] = close.pct_change()

    # 成交量比 (相对 20 周期均线)
    vol_ma = volume.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = np.log(volume / (vol_ma + 1e-8))

    # 持仓量变化
    if "oi" in df.columns:
        oi = df["oi"].astype(float)
        df["oi_change"] = oi.pct_change()
    else:
        df["oi_change"] = 0.0

    # ---- 多尺度 return (4) ----
    for w in RETURN_WINDOWS:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = close / close.shift(w)
            ratio = ratio.replace(0, np.nan)  # 防止 log(0)
            df[f"return_{w}"] = np.log(ratio)

    # ---- 多尺度 volatility (4) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(close / close.shift(1))
    for w in VOL_WINDOWS:
        df[f"volatility_{w}"] = ret.rolling(w, min_periods=1).std()

    # ---- 多尺度 MA_gap (4) ----
    for w in MA_WINDOWS:
        ma = close.rolling(w, min_periods=1).mean()
        df[f"ma_gap_{w}"] = (close - ma) / (ma + 1e-8)

    # ---- 时间特征 (4) ----
    dt = df["datetime"]
    hour = dt.dt.hour
    minute = dt.dt.minute

    # 一天内的分钟编号 (简化: 0 到 ~390)
    minutes_in_day = hour * 60 + minute
    df["minute_of_session"] = minutes_in_day / 1440.0  # 归一化到 [0, 1]

    # session_progress: 当日内的进度 (用 cumulative count by date)
    dates = dt.dt.date
    df["session_progress"] = df.groupby(dates).cumcount()
    max_count = df.groupby(dates)["session_progress"].transform("max") + 1
    df["session_progress"] = df["session_progress"] / max_count

    # 周期编码
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # 选择输出列
    feature_cols = [
        "norm_open", "high_pct", "low_pct", "minute_return",
        "volume_ratio", "oi_change",
        *[f"return_{w}" for w in RETURN_WINDOWS],
        *[f"volatility_{w}" for w in VOL_WINDOWS],
        *[f"ma_gap_{w}" for w in MA_WINDOWS],
        "minute_of_session", "session_progress", "hour_sin", "hour_cos",
    ]
    assert len(feature_cols) == NUM_MARKET_FEATURES

    output = df[["datetime"] + feature_cols].copy()
    return output


def compute_train_stats(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1):
    """用训练集计算归一化统计量, 返回 (stats_dict, train_end_idx, val_end_idx)"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    feature_cols = [c for c in df.columns if c != "datetime"]
    train_data = df.iloc[:train_end][feature_cols].values.astype(np.float32)

    # 跳过前 max_window 行 (有 NaN)
    max_window = max(max(VOL_WINDOWS), max(MA_WINDOWS), max(RETURN_WINDOWS))
    train_data = train_data[max_window:]

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train_data, axis=0)
        std = np.nanstd(train_data, axis=0)
    std[std < 1e-8] = 1.0  # 防止除零

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "num_features": len(feature_cols),
        "total_rows": n,
    }
    return stats


def preprocess_one(symbol: str):
    """处理单个品种"""
    info = FUTURES[symbol]
    input_path = os.path.join(INPUT_DIR, f"{symbol}.csv")
    if not os.path.exists(input_path):
        print(f"  跳过 {symbol}: {input_path} 不存在")
        return

    print(f"\n=== {info['name']} ({symbol}) ===")
    df = pd.read_csv(input_path)
    print(f"  原始数据: {len(df)} 条")

    # 计算特征
    df_features = compute_features(df, symbol)

    # 删除前 max_window 行 (NaN)
    max_window = max(max(VOL_WINDOWS), max(MA_WINDOWS), max(RETURN_WINDOWS))
    df_features = df_features.iloc[max_window:].reset_index(drop=True)

    # 检查并修复 NaN / inf
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    nan_counts = df_features.isnull().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        print(f"  警告: 存在 NaN/inf ({total_nan} 个), 填充为 0")
    df_features = df_features.fillna(0)

    # 计算训练集归一化统计量
    stats = compute_train_stats(df_features)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    out_json = os.path.join(OUTPUT_DIR, f"{symbol}_stats.json")

    df_features.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    feature_cols = [c for c in df_features.columns if c != "datetime"]
    print(f"  特征数: {len(feature_cols)}")
    print(f"  数据行: {len(df_features)}")
    print(f"  时间: {df_features['datetime'].iloc[0]} ~ {df_features['datetime'].iloc[-1]}")
    print(f"  训练集: 0 ~ {stats['train_end_idx']}")
    print(f"  验证集: {stats['train_end_idx']} ~ {stats['val_end_idx']}")
    print(f"  测试集: {stats['val_end_idx']} ~ {len(df_features)}")
    print(f"  保存: {out_csv}")
    print(f"  统计: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="期货分钟级数据特征工程")
    parser.add_argument("--symbol", type=str, default="all",
                        help="品种代码 (RB/I/AU/CU/M), 默认全部")
    args = parser.parse_args()

    symbols = list(FUTURES.keys()) if args.symbol == "all" else [args.symbol]
    for s in symbols:
        preprocess_one(s)

    print(f"\n完成!")


if __name__ == "__main__":
    main()
