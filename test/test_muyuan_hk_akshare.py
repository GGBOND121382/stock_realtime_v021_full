#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp
import traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


CANDIDATES = ["02714", "2714", "HK2714", "2714.HK"]
OUT_DIR = Path("debug_hk_muyuan")
TIMEOUT = 20


def run_with_timeout(name, fn, timeout=TIMEOUT):
    def worker(q):
        try:
            df = fn()
            if df is None:
                q.put({"ok": False, "error": "returned None"})
                return
            if not isinstance(df, pd.DataFrame):
                q.put({"ok": False, "error": f"returned {type(df)}"})
                return
            q.put({
                "ok": True,
                "shape": df.shape,
                "columns": list(df.columns),
                "head": df.head(5).astype(str).to_dict(orient="records"),
                "tail": df.tail(5).astype(str).to_dict(orient="records"),
            })
        except Exception as e:
            q.put({
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=5),
            })

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"name": name, "ok": False, "error": f"timeout>{timeout}s"}

    if q.empty():
        return {"name": name, "ok": False, "error": "no result from child process"}

    r = q.get()
    r["name"] = name
    return r


def save_df(name, df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def main():
    import akshare as ak

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = (date.today() - timedelta(days=120)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    start_year = str(date.today().year - 1)
    end_year = str(date.today().year)

    results = []

    # 1. A+H 实时全表：最适合盘中实时信号参考
    def test_ah_spot():
        df = ak.stock_zh_ah_spot()
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            s = df[col].astype(str)
            mask |= s.str.contains("02714|2714|牧原", regex=True, na=False)
        hit = df[mask].copy()
        save_df("stock_zh_ah_spot_hit", hit)
        return hit

    results.append(run_with_timeout("stock_zh_ah_spot_find_muyuan", test_ah_spot))

    # 2. 港股东财实时全表：找 02714 / 2714 / 牧原
    def test_hk_spot_em():
        df = ak.stock_hk_spot_em()
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            s = df[col].astype(str)
            mask |= s.str.contains("02714|2714|牧原", regex=True, na=False)
        hit = df[mask].copy()
        save_df("stock_hk_spot_em_hit", hit)
        return hit

    results.append(run_with_timeout("stock_hk_spot_em_find_muyuan", test_hk_spot_em))

    # 3. A+H 历史日线：按候选代码逐个测
    for code in CANDIDATES:
        def fn(code=code):
            df = ak.stock_zh_ah_daily(
                symbol=code,
                start_year=start_year,
                end_year=end_year,
                adjust="",
            )
            save_df(f"stock_zh_ah_daily_{code.replace('.', '_')}", df)
            return df

        results.append(run_with_timeout(f"stock_zh_ah_daily({code})", fn))

    # 4. 港股历史日线-东财：按候选代码逐个测
    for code in CANDIDATES:
        def fn(code=code):
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="",
            )
            save_df(f"stock_hk_hist_{code.replace('.', '_')}", df)
            return df

        results.append(run_with_timeout(f"stock_hk_hist({code})", fn))

    # 5. 如有新浪港股日线接口，也测一下。不同 AKShare 版本可能没有这个函数。
    if hasattr(ak, "stock_hk_daily"):
        for code in CANDIDATES:
            def fn(code=code):
                df = ak.stock_hk_daily(symbol=code)
                save_df(f"stock_hk_daily_{code.replace('.', '_')}", df)
                return df

            results.append(run_with_timeout(f"stock_hk_daily({code})", fn))
    else:
        results.append({
            "name": "stock_hk_daily",
            "ok": False,
            "error": "akshare has no function stock_hk_daily",
        })

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for r in results:
        status = "OK" if r.get("ok") and tuple(r.get("shape", (0, 0)))[0] > 0 else "FAIL"
        print(f"{status:4s} | {r['name']}")
        if r.get("shape"):
            print(f"     shape={r['shape']}, columns={r.get('columns')}")
        if r.get("error"):
            print(f"     error={r['error']}")
        print()

    print(f"Saved details to: {summary_path}")


if __name__ == "__main__":
    main()