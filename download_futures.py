"""
下载国内主流商品期货主力连续合约日线数据（Tushare Pro）
原始数据保存到 dataset/futures_origin/
用法: python download_futures.py
"""

import os
import time
import tushare as ts
import pandas as pd

TOKEN = "8347a171dac7fe6fc71fdb12e9455a424754db51efc025b5b9acdc63"

# 下载品种配置
FUTURES = {
    "RB": {"ts_code": "RB.SHF", "name": "螺纹钢"},
    "I":  {"ts_code": "I.DCE",  "name": "铁矿石"},
    "AU": {"ts_code": "AU.SHF", "name": "黄金"},
    "CU": {"ts_code": "CU.SHF", "name": "铜"},
    "M":  {"ts_code": "M.DCE",  "name": "豆粕"},
}

START_DATE = "20150101"
END_DATE = "20260523"
OUTPUT_DIR = "./dataset/futures_origin"


def download_one(pro, ts_code, symbol, name):
    """分页下载单个品种的日线数据"""
    print(f"\n--- 下载 {name} ({symbol}) ---")

    # 按年分段请求，避免单次超过 2000 条限制
    all_dfs = []
    years = range(2015, 2027)
    for y in years:
        s = f"{y}0101"
        e = f"{y}1231"
        if e > END_DATE:
            e = END_DATE
        try:
            df = pro.fut_daily(ts_code=ts_code, start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f"  {y}: {len(df)} 条")
            else:
                print(f"  {y}: 无数据")
        except Exception as ex:
            print(f"  {y}: 请求失败 - {ex}")
        time.sleep(0.3)  # 避免 frequency limit

    if not all_dfs:
        print(f"  {name} 无任何数据，跳过")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    # 转为项目所需格式
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # 保留有用字段
    cols = ["date", "open", "high", "low", "close", "settle", "vol", "amount", "oi"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"  共 {len(df)} 条, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    return df


def main():
    pro = ts.pro_api(TOKEN)

    # 测试连接
    try:
        test = pro.trade_cal(exchange="DCE", start_date="20260101", end_date="20260110")
        print(f"Tushare 连接成功，积分余额查询通过")
    except Exception as e:
        print(f"Tushare 连接失败: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol, info in FUTURES.items():
        df = download_one(pro, info["ts_code"], symbol, info["name"])
        if df is not None:
            path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"  已保存: {path}")

    print(f"\n全部完成！文件保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
