#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp
import traceback
from pathlib import Path

import pandas as pd
import akshare as ak


CODES = {
    "01610": "cofco_joycome",
    "00288": "wh_group",
    "01068": "yurun_food",
    "01117": "modern_dairy",
}

OUT = Path("debug_hk_proxy_daily")
OUT.mkdir(exist_ok=True)


def run_with_timeout(name, fn, timeout=25):
    def worker(q):
        try:
            df = fn()
            if not isinstance(df, pd.DataFrame):
                q.put({"ok": False, "error": f"not dataframe: {type(df)}"})
                return
            q.put({
                "ok": True,
                "shape": df.shape,
                "columns": list(df.columns),
                "head": df.head(3).astype(str).to_dict("records"),
                "tail": df.tail(3).astype(str).to_dict("records"),
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
        return {"name": name, "ok": False, "error": "no child result"}

    r = q.get()
    r["name"] = name
    return r


def save_df(name, df):
    p = OUT / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")


def main():
    results = []

    for code, label in CODES.items():
        # 1. 首选：新浪港股日线。牧原 02714 已验证这个接口可用。
        def fn_daily(code=code, label=label):
            df = ak.stock_hk_daily(symbol=code)
            save_df(f"stock_hk_daily_{code}_{label}", df)
            return df

        results.append(run_with_timeout(f"stock_hk_daily({code})/{label}", fn_daily))

        # 2. 对照：东财港股历史。之前牧原就是这个接口失败。
        def fn_hist(code=code, label=label):
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                adjust="",
            )
            save_df(f"stock_hk_hist_{code}_{label}", df)
            return df

        results.append(run_with_timeout(f"stock_hk_hist({code})/{label}", fn_hist))

    (OUT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for r in results:
        ok = r.get("ok") and r.get("shape", [0])[0] > 0
        print(("OK  " if ok else "FAIL"), r["name"])
        if r.get("shape"):
            print("    shape:", r["shape"])
            print("    cols:", r.get("columns"))
        if r.get("error"):
            print("    error:", r["error"])
        print()

    print("saved:", OUT / "summary.json")


if __name__ == "__main__":
    main()