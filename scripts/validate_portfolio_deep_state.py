#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, importlib, json, subprocess, sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

def run(cmd):
    print('[RUN]', ' '.join(cmd)); return subprocess.run(cmd,cwd=PROJECT_DIR).returncode==0

def fail(errors,msg): print('[ERROR]',msg); errors.append(msg)
def ok(msg): print('[OK]',msg)

def normalize_stock_code(x:Any)->str:
    s=str(x or '').strip().upper()
    if not s or s in {'NAN','NONE','NULL'}: return ''
    if s.isdigit() and len(s)==6: return f'{s}.SH' if s.startswith(('5','6','9')) else f'{s}.SZ'
    if s.startswith('SH.'): return f'{s[3:]}.SH'
    if s.startswith('SZ.'): return f'{s[3:]}.SZ'
    return s

def resolve_path(raw:Any, stock_code:str='')->Optional[Path]:
    text=str(raw or '').strip().replace('\\','/')
    if not text or text.lower() in {'nan','none','null'}: return None
    p=Path(text)
    if p.exists(): return p
    if not p.is_absolute() and (PROJECT_DIR/p).exists(): return PROJECT_DIR/p
    for marker in ['stock_realtime_v021_full/','stock_realtime/']:
        if marker in text:
            q=PROJECT_DIR/text.split(marker,1)[1]
            if q.exists(): return q
    if 'saved_data/' in text:
        q=PROJECT_DIR/text[text.index('saved_data/'):]
        if q.exists(): return q
    name=Path(text).name; code=normalize_stock_code(stock_code).split('.',1)[0] if stock_code else ''
    roots=[]
    if code: roots.append(PROJECT_DIR/'saved_data'/f'{code}_pipeline_out')
    roots.append(PROJECT_DIR/'saved_data')
    for root in roots:
        if root.exists() and name:
            hits=list(root.rglob(name))
            if hits: return hits[0]
    return None

def load_json(path:Path)->dict:
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return {}

def check_markers(errors):
    files={
        'backtest': PROJECT_DIR/'portfolio_decision/backtest_historical_score_portfolio.py',
        'adapter': PROJECT_DIR/'portfolio_decision/portfolio_confirm_from_buy_signals.py',
        'optimizer': PROJECT_DIR/'portfolio_decision/daily_portfolio_confirm_pyscipopt.py',
        'audit': PROJECT_DIR/'scripts/settle_portfolio_backtest_consistently.py',
    }
    for name,path in files.items():
        if not path.exists(): fail(errors,f'missing {name}: {path}')
        else: ok(f'found {name}: {path}')
    checks=[
        (files['backtest'],['from model_training.optimize_nextday_vwap_model import add_trade_returns','samples = add_trade_returns(','buy_mask = (all_scores["signal"] == True)','entry_policy=lot.entry_policy','expected_return_col=lot.expected_return_col']),
        (files['adapter'],['def as_text(','"samples": samples','"expected_return_col": expected_return_col','"entry_vwap_premium_bps": entry_vwap_premium_bps']),
        (files['optimizer'],['entry_policy: str','entry_vwap_premium_bps: float','samples: str','expected_return_col: str','metadata_path: str','entry_policy=entry_policy','expected_return_col=expected_return_col']),
    ]
    for path,needles in checks:
        if not path.exists(): continue
        txt=path.read_text(encoding='utf-8'); missing=[n for n in needles if n not in txt]
        if missing: fail(errors,f'{path} missing markers: {missing}')
        else: ok(f'markers present: {path}')

def check_compile_import(errors):
    files=['portfolio_decision/backtest_historical_score_portfolio.py','portfolio_decision/portfolio_confirm_from_buy_signals.py','portfolio_decision/daily_portfolio_confirm_pyscipopt.py','scripts/settle_portfolio_backtest_consistently.py']
    for f in files:
        if (PROJECT_DIR/f).exists():
            if run([sys.executable,'-m','py_compile',f]): ok(f'py_compile {f}')
            else: fail(errors,f'py_compile failed: {f}')
    try:
        mod=importlib.import_module('portfolio_decision.backtest_historical_score_portfolio'); ok('import backtest module')
    except Exception as exc:
        fail(errors,f'import backtest failed: {type(exc).__name__}: {exc}'); return
    for cls_name in ['OpenLot','TradeRecord']:
        cls=getattr(mod,cls_name,None)
        if cls is None or not is_dataclass(cls): fail(errors,f'{cls_name} missing or not dataclass'); continue
        seen_default=False; bad=[]
        for f in fields(cls):
            has_default=not (f.default.__class__.__name__=='_MISSING_TYPE' and f.default_factory.__class__.__name__=='_MISSING_TYPE')
            if has_default: seen_default=True
            elif seen_default: bad.append(f.name)
        if bad: fail(errors,f'{cls_name} non-default fields after default fields: {bad}')
        else: ok(f'{cls_name} dataclass field order')

def check_sample_generation(errors,max_artifacts:int):
    try: from model_training.optimize_nextday_vwap_model import add_trade_returns
    except Exception as exc: fail(errors,f'cannot import add_trade_returns: {type(exc).__name__}: {exc}'); return
    metas=sorted((PROJECT_DIR/'saved_models').glob('*/*/metadata.json'))
    if not metas: print('[WARN] no saved_models metadata found'); return
    checked=0
    for meta_path in metas:
        if checked>=max_artifacts: break
        meta=load_json(meta_path); stock=normalize_stock_code(meta.get('stock_code') or meta_path.parent.parent.name); sample_path=resolve_path(meta.get('samples'),stock)
        if sample_path is None or not sample_path.exists(): fail(errors,f'samples missing: {stock} {meta_path.parent.name} -> {meta.get("samples")}'); continue
        try: df=pd.read_csv(sample_path,nrows=200)
        except Exception as exc: fail(errors,f'read samples failed {sample_path}: {type(exc).__name__}: {exc}'); continue
        if 'date' not in df.columns: fail(errors,f'samples missing date: {sample_path}'); continue
        try:
            out=add_trade_returns(df,cost_bps=float(meta.get('round_trip_cost_bps',1.7) or 1.7),target_bps=float(meta.get('target_hit_bps',50.0) or 50.0),entry_policy=str(meta.get('entry_policy') or 'vwap_low'),entry_vwap_premium_bps=float(meta.get('entry_vwap_premium_bps',50.0) or 50.0))
        except Exception as exc: fail(errors,f'add_trade_returns failed {stock} {meta_path.parent.name}: {type(exc).__name__}: {exc}'); continue
        required=['entry_signal','trade_net_close_return','trade_net_high_return','trade_hit_label','trade_target_or_close_return','trade_close_profit_label']
        missing=[c for c in required if c not in out.columns]
        if missing: fail(errors,f'add_trade_returns missing columns {missing}: {stock} {meta_path.parent.name}'); continue
        checked+=1
    if checked: ok(f'sample add_trade_returns checks passed for {checked} artifacts')
    else: fail(errors,'no artifact sample passed add_trade_returns check')

def check_existing_outputs(errors,backtest_dir:Path):
    if not backtest_dir.exists(): print(f'[WARN] no backtest dir yet: {backtest_dir}'); return
    inputs=list(backtest_dir.glob('portfolio_runs/*/_portfolio_inputs_*/portfolio_signals.csv'))
    for p in inputs[:20]:
        try: df=pd.read_csv(p)
        except Exception: continue
        for c in ['entry_policy','samples','expected_return_col']:
            if c not in df.columns: fail(errors,f'{p}: missing {c}')
            elif len(df) and df[c].isna().all(): fail(errors,f'{p}: {c} all NaN')
    if inputs: ok(f'checked portfolio_signals files: {len(inputs[:20])}')
    trades=backtest_dir/'historical_score_portfolio_backtest_trades.csv'
    if trades.exists():
        df=pd.read_csv(trades)
        for c in ['entry_policy','samples','expected_return_col','metadata_path']:
            if c not in df.columns: fail(errors,f'trades missing {c}')
            elif len(df) and df[c].isna().all(): fail(errors,f'trades {c} all NaN')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-artifacts',type=int,default=20); ap.add_argument('--backtest-dir',default='portfolio_reports/backtests/historical_score_portfolio'); ap.add_argument('--skip-sample-check',action='store_true')
    args=ap.parse_args(); errors=[]
    check_markers(errors); check_compile_import(errors)
    if not args.skip_sample_check: check_sample_generation(errors,args.max_artifacts)
    check_existing_outputs(errors,PROJECT_DIR/args.backtest_dir)
    print('='*72)
    if errors:
        print(f'[FAIL] {len(errors)} issue(s)')
        for i,e in enumerate(errors,1): print(f'{i}. {e}')
        return 2
    print('[OK] no syntax/import/static/model-usage issues found')
    return 0
if __name__=='__main__': raise SystemExit(main())
