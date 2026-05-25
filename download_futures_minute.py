"""
下载国内主流商品期货主力连续合约 1 分钟 K 线数据 (RQData)
原始数据保存到 dataset/futures_origin_minute/
用法:
    python download_futures_minute.py --symbol RB --start 20150101
    python download_futures_minute.py --resume   # 断点续传
"""

import os
import argparse
import pandas as pd

# 品种配置
FUTURES = {
    "RB": {"name": "螺纹钢"},
    "I":  {"name": "铁矿石"},
    "AU": {"name": "黄金"},
    "CU": {"name": "铜"},
    "M":  {"name": "豆粕"},
}

OUTPUT_DIR = "./dataset/futures_origin_minute"

RQDATA_PASSWORD = (
    "i2MuZjp3s-mC3k_j-HE3mpDPTb2Ffeuga5-9lk5QjpIVDl8Q3-6uG-6Gsx488Uv5"
    "OGKKw8b__rI10_hRusJd2H9FtKsCdopOIwFYPX077lZuIVkJjsnc1HB-DFy1U9kc41"
    "pv4ILNulOehAgKY7W5FzVVPdEjiU_37Sqnq2HsENk=RI36TU8pqdK_G5SIPjpUiarL"
    "7oqWsPeplCAJq-TcQ7Mv_zt6TK-ysq-rKNcnwj9fJLzUPAGpPtfCAv1OtJEYudqTft"
    "wHpi7ACp3UCSGc8rnBFaX2umYNRVFOJCV3T56srm_ZZd_w3pd-83DdH-DL5u4SYHJu"
    "oKJZVHnSegYHFN8="
)


def _init_rqdata():
    """初始化 rqdatac 连接"""
    import rqdatac as rq
    rq.init("license", RQDATA_PASSWORD)
    return rq


def _generate_month_ranges(start_date: str, end_date: str):
    """生成按月的 (start, end) 范围列表"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    ranges = []
    current = start.replace(day=1)
    while current <= end:
        month_end = (current + pd.DateOffset(months=1) - pd.Timedelta(days=1))
        if month_end > end:
            month_end = end
        ranges.append((
            current.strftime("%Y%m%d"),
            month_end.strftime("%Y%m%d"),
        ))
        current = current + pd.DateOffset(months=1)
    return ranges


def download_minute_rqdata(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """使用 RQData 下载期货主力连续合约 1 分钟 K 线 (按月分批)"""
    rq = _init_rqdata()

    month_ranges = _generate_month_ranges(start_date, end_date)
    all_dfs = []

    for i, (m_start, m_end) in enumerate(month_ranges):
        print(f"  [{i+1}/{len(month_ranges)}] {m_start} ~ {m_end} ...", end=" ", flush=True)
        try:
            df = rq.futures.get_dominant_price(
                underlying_symbols=symbol,
                start_date=m_start,
                end_date=m_end,
                frequency="1m",
                fields=None,
                adjust_type="none",
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f"{len(df)} 条")
            else:
                print("无数据")
        except Exception as ex:
            print(f"失败: {ex}")

    if not all_dfs:
        print(f"  {symbol} 无任何数据返回")
        return None

    df = pd.concat(all_dfs)
    return df


def process_rqdata_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 RQData 返回的 MultiIndex DataFrame 转为统一格式"""
    df = df.reset_index()

    # 列名映射
    rename = {
        "total_turnover": "amount",
        "open_interest": "oi",
    }
    df = df.rename(columns=rename)

    # 确保 datetime 列存在
    if "datetime" not in df.columns:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.rename(columns={col: "datetime"})
                break

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # 只保留标准列
    keep = ["datetime", "open", "high", "low", "close", "volume", "amount", "oi"]
    df = df[[c for c in keep if c in df.columns]]

    # 确保数值类型
    for c in ["open", "high", "low", "close", "volume", "amount", "oi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def check_resume(symbol: str) -> str:
    """检查已有 CSV 最后日期，返回续传起始日期 (空字符串表示从头开始)"""
    path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return ""
    try:
        df_tail = pd.read_csv(path)
        last_dt = pd.to_datetime(df_tail["datetime"].iloc[-1])
        resume_date = (last_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
        print(f"  已有数据截至 {last_dt}, 从 {resume_date} 续传")
        return resume_date
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="下载期货分钟级数据 (RQData)")
    parser.add_argument("--symbol", type=str, default="all",
                        help="品种代码 (RB/I/AU/CU/M), 默认全部")
    parser.add_argument("--start", type=str, default="20150101",
                        help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260523",
                        help="结束日期 YYYYMMDD")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 从已有 CSV 最后日期继续")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    symbols = list(FUTURES.keys()) if args.symbol == "all" else [args.symbol]

    for symbol in symbols:
        print(f"\n=== {FUTURES[symbol]['name']} ({symbol}) ===")

        start_date = args.start
        end_date = args.end

        # 断点续传
        if args.resume:
            resume_start = check_resume(symbol)
            if resume_start and resume_start >= end_date:
                print(f"  数据已是最新, 跳过")
                continue
            if resume_start:
                start_date = resume_start

        # 下载
        df = download_minute_rqdata(symbol, start_date, end_date)
        if df is None:
            continue
        df = process_rqdata_df(df)

        # 追加或覆盖保存
        path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
        if args.resume and os.path.exists(path):
            old_df = pd.read_csv(path)
            old_df["datetime"] = pd.to_datetime(old_df["datetime"])
            df = pd.concat([old_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        df.to_csv(path, index=False)
        print(f"  保存 {len(df)} 条 -> {path}")
        print(f"  时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    print(f"\n完成! 文件保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
