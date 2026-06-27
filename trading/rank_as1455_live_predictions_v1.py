#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank AS1455 live predictions."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def normalize_symbol(value: object) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan": return ""
    s = s.replace(".XSHE",".SZ").replace(".XSHG",".SH")
    m = re.search(r"(\d{6})", s)
    if m: code = m.group(1)
    elif re.fullmatch(r"\d{1,6}", s): code = s.zfill(6)
    else: return s.upper()
    return f"{code}.SH" if code.startswith(("6","9")) else f"{code}.SZ"

def compact_symbol(symbol: str) -> str:
    m = re.search(r"(\d{6})", str(symbol))
    return m.group(1) if m else str(symbol)

def infer_board(symbol: str) -> str:
    code=compact_symbol(symbol)
    if code.startswith(("600","601","603","605")): return "sh_mainboard"
    if code.startswith(("000","001","002","003")): return "sz_mainboard"
    if code.startswith(("300","301")): return "chinext"
    if code.startswith(("688","689")): return "star"
    if code.startswith(("8","4","920")): return "bse"
    return "unknown"

def json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--out-rank", default=None)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--mainboard-only", action="store_true", default=True)
    args=ap.parse_args()
    start=time.time()
    live_dir=Path(args.live_dir)
    pred_path=Path(args.predictions) if args.predictions else live_dir/"14_live_predictions.csv"
    out_rank=Path(args.out_rank) if args.out_rank else live_dir/"15_live_rank.csv"
    out_report=Path(args.out_report) if args.out_report else live_dir/"15_live_rank_report.json"
    report={"passed":False,"predictions":str(pred_path)}
    try:
        df=pd.read_csv(pred_path)
        req={"date","symbol","pred_score"}
        miss=sorted(req-set(df.columns))
        if miss: raise RuntimeError(f"predictions missing columns: {miss}")
        df=df.copy()
        df["symbol"]=df["symbol"].map(normalize_symbol)
        df["pred_score"]=pd.to_numeric(df["pred_score"], errors="coerce")
        df=df.dropna(subset=["date","symbol","pred_score"])
        df=df[df["symbol"].astype(str).str.len()>0]
        if df.empty: raise RuntimeError("empty predictions after cleaning")
        df=df.sort_values(["date","pred_score","symbol"], ascending=[True,False,True]).copy()
        df["rank"]=df.groupby("date")["pred_score"].rank(method="first", ascending=False).astype(int)
        df["board"]=df["symbol"].map(infer_board)
        df["is_mainboard"]=df["board"].isin(["sh_mainboard","sz_mainboard"])
        df["trade_allowed_mainboard"]=df["is_mainboard"]
        if args.universe:
            uni_path=Path(args.universe)
        else:
            uni_path=live_dir/"01_universe.csv"
        if uni_path.exists():
            uni=pd.read_csv(uni_path)
            if "symbol" not in uni.columns and "code" in uni.columns:
                uni["symbol"]=uni["code"]
            if "symbol" in uni.columns:
                uni["symbol"]=uni["symbol"].map(normalize_symbol)
                keep_cols=[c for c in ["symbol","name","board","industry"] if c in uni.columns]
                uni=uni[keep_cols].drop_duplicates("symbol")
                df=df.merge(uni, on="symbol", how="left", suffixes=("","_universe"))
                if "board_universe" in df.columns:
                    df["board"]=df["board_universe"].where(df["board_universe"].notna(), df["board"])
                    df.drop(columns=["board_universe"], inplace=True)
                    df["is_mainboard"]=df["board"].isin(["sh_mainboard","sz_mainboard"])
                    df["trade_allowed_mainboard"]=df["is_mainboard"]
        out_rank.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_rank,index=False,encoding="utf-8-sig")
        report.update({"passed":True,"rank_file":str(out_rank),"rows":int(len(df)),"symbols":int(df.symbol.nunique()),"date_min":str(df.date.min()),"date_max":str(df.date.max()),"top1":df.sort_values("rank").head(1).to_dict("records")})
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        out_report.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"]=round(time.time()-start,3)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=json_default),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"rank_file":report.get("rank_file"),"rows":report.get("rows")},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
