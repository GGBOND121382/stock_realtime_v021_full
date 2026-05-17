#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from model_training.optimize_nextday_vwap_model import add_trade_returns


def norm_text(x: Any) -> str:
    s = str(x or '').strip()
    return '' if s.lower() in {'nan','none','null'} else s

def as_float(x: Any, default: float=np.nan) -> float:
    try:
        if x is None or pd.isna(x): return default
    except Exception: pass
    try: return float(x)
    except Exception: return default

def date_key(x: Any) -> str:
    s = norm_text(x)
    if not s: return ''
    try:
        if s.isdigit() and len(s)==8: return pd.to_datetime(s, format='%Y%m%d').strftime('%Y-%m-%d')
        return pd.to_datetime(s).strftime('%Y-%m-%d')
    except Exception: return s[:10]

def normalize_stock_code(x: Any) -> str:
    s = norm_text(x).upper()
    if not s: return ''
    if s.isdigit() and len(s)==6: return f"{s}.SH" if s.startswith(('5','6','9')) else f"{s}.SZ"
    if s.startswith('SH.'): return f"{s[3:]}.SH"
    if s.startswith('SZ.'): return f"{s[3:]}.SZ"
    return s

def resolve_path(x: Any, stock: str='') -> Optional[Path]:
    s = norm_text(x).replace('\\','/')
    if not s: return None
    p = Path(s)
    if p.exists(): return p
    if not p.is_absolute() and (PROJECT_DIR / p).exists(): return PROJECT_DIR / p
    for marker in ['stock_realtime_v021_full/','stock_realtime/']:
        if marker in s:
            q = PROJECT_DIR / s.split(marker,1)[1]
            if q.exists(): return q
    if 'saved_data/' in s:
        q = PROJECT_DIR / s[s.index('saved_data/'):]
        if q.exists(): return q
    name = Path(s).name
    code = normalize_stock_code(stock).split('.',1)[0] if stock else ''
    roots = []
    if code: roots.append(PROJECT_DIR/'saved_data'/f'{code}_pipeline_out')
    roots.append(PROJECT_DIR/'saved_data')
    for r in roots:
        if r.exists() and name:
            hits = list(r.rglob(name))
            if hits: return hits[0]
    return p if p.is_absolute() else PROJECT_DIR/p

def read_json(p: Optional[Path]) -> dict:
    if not p or not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}

def find_metadata(saved_models: Path, stock: str, artifact: str, explicit: Any=None) -> Optional[Path]:
    p = resolve_path(explicit, stock)
    if p and p.exists(): return p
    if not saved_models.is_absolute(): saved_models = PROJECT_DIR / saved_models
    direct = saved_models / stock / artifact / 'metadata.json'
    if direct.exists(): return direct
    hits = list(saved_models.glob(f'*/{artifact}/metadata.json'))
    return hits[0] if hits else None

def norm_entry(x: str) -> str:
    v = str(x or '').strip().lower().replace('-','_')
    return {'default':'vwap_low','candidate':'vwap_low','low_vwap':'vwap_low','below_vwap':'vwap_low','all':'all_days','all_day':'all_days','all_dates':'all_days','full':'all_days'}.get(v,v)

def norm_label(x: str) -> str:
    s = str(x or '').strip().lower()
    if s.startswith('hit'): return 'hit'
    if s in {'close','close_profit','profit'}: return 'close_profit'
    return s

def expected_col(label: str) -> str:
    label = norm_label(label)
    if label == 'hit': return 'trade_target_or_close_return'
    if label == 'close_profit': return 'trade_net_close_return'
    return ''

def infer_entry(sig, meta, tr):
    for src,obj in [('trade',tr),('signal',sig)]:
        if obj is None: continue
        for c in ['entry_policy','signal_entry_policy']:
            if c in obj.index:
                v = norm_entry(norm_text(obj.get(c)))
                if v: return v, f'{src}:{c}'
    v = norm_entry(norm_text(meta.get('entry_policy')))
    if v: return v, 'metadata'
    text = ' '.join(norm_text(obj.get(c)).lower() for obj in [sig,tr] if obj is not None for c in ['artifact_name','model_name'] if c in obj.index)
    if 'vwap_low' in text: return 'vwap_low','artifact_name'
    if 'all_days' in text: return 'all_days','artifact_name'
    return '','missing'

def infer_label(sig, meta, tr):
    for src,obj in [('trade',tr),('signal',sig)]:
        if obj is None: continue
        for c in ['label_mode','signal_label_mode']:
            if c in obj.index:
                v = norm_label(norm_text(obj.get(c)))
                if v: return v, f'{src}:{c}'
    v = norm_label(norm_text(meta.get('label_mode')))
    if v: return v, 'metadata'
    text = ' '.join(norm_text(obj.get(c)).lower() for obj in [sig,tr] if obj is not None for c in ['artifact_name','model_name'] if c in obj.index)
    if 'hit' in text: return 'hit','artifact_name'
    if 'close_profit' in text: return 'close_profit','artifact_name'
    return '','missing'

def read_signals(bt: Path, date: str) -> pd.DataFrame:
    ymd = pd.to_datetime(date).strftime('%Y%m%d')
    frames=[]
    for fn,prio in [('all_scores.csv',0),('buy_signals.csv',1)]:
        p = bt/'generated_signals'/ymd/fn
        if not p.exists(): continue
        try: df = pd.read_csv(p)
        except Exception: continue
        df['_signal_priority']=prio
        if 'stock_code' in df.columns: df['stock_code']=df['stock_code'].map(normalize_stock_code)
        df['_date_key']=df['date'].map(date_key) if 'date' in df.columns else date_key(date)
        frames.append(df)
    if not frames: return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True).sort_values('_signal_priority')
    subset=[c for c in ['stock_code','artifact_name','model_name','_date_key'] if c in out.columns]
    if subset: out=out.drop_duplicates(subset=subset,keep='first')
    return out.reset_index(drop=True)

def match_signal(signals: pd.DataFrame, tr):
    if signals.empty: return None,'SIGNAL_FILE_MISSING'
    stock=normalize_stock_code(tr.get('stock_code')); model=norm_text(tr.get('model_name')); d=date_key(tr.get('buy_date'))
    df=signals.copy()
    if 'stock_code' in df.columns: df=df[df['stock_code']==stock]
    if '_date_key' in df.columns: df=df[df['_date_key']==d]
    if df.empty: return None,'SIGNAL_ROW_MISSING_STOCK_DATE'
    if 'artifact_name' in df.columns:
        hit=df[df['artifact_name'].astype(str)==model]
        if not hit.empty: return hit.iloc[0],'OK_ARTIFACT_NAME_MATCH'
    if 'model_name' in df.columns:
        hit=df[df['model_name'].astype(str)==model]
        if not hit.empty: return hit.iloc[0],'OK_MODEL_NAME_MATCH'
    return df.iloc[0],'ARTIFACT_MISMATCH_FALLBACK'

def choose_sample(sig, meta, stock):
    if sig is not None:
        for c in ['samples','sample_file','samples_path','training_samples']:
            if c in sig.index:
                p=resolve_path(sig.get(c),stock)
                if p and p.exists(): return p
    for c in ['samples','sample_file','samples_path','training_samples']:
        p=resolve_path(meta.get(c),stock)
        if p and p.exists(): return p
    return None

def boolish(x):
    try:
        if pd.isna(x): return None
    except Exception: pass
    if isinstance(x,(bool,np.bool_)): return bool(x)
    s=str(x).strip().lower()
    if s in {'1','true','t','yes','y'}: return True
    if s in {'0','false','f','no','n'}: return False
    return None

def load_trade_samples(path, cache, cost, target, entry, premium):
    key=(str(path),round(cost,8),round(target,8),norm_entry(entry),round(premium,8))
    if key in cache: return cache[key]
    df=pd.read_csv(path)
    date_col=next((c for c in ['date','trade_date','datetime','time'] if c in df.columns),None)
    if date_col is None: cache[key]=None; return None
    df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
    df['_date_key']=df[date_col].map(date_key)
    df=add_trade_returns(df,cost_bps=cost,target_bps=target,entry_policy=norm_entry(entry),entry_vwap_premium_bps=premium)
    cache[key]=df
    return df

def pf(pnl):
    pnl=pd.to_numeric(pnl,errors='coerce').dropna(); gain=float(pnl[pnl>0].sum()); loss=float(-pnl[pnl<0].sum())
    if loss<=0: return float('inf') if gain>0 else 0.0
    return gain/loss

def comp_dd(ret):
    r=pd.to_numeric(ret,errors='coerce').dropna().to_numpy(float)
    if len(r)==0: return 0.0
    eq=np.cumprod(1+r); return float(np.min(eq/np.maximum.accumulate(eq)-1))

def pnl_dd(pnl):
    x=pd.to_numeric(pnl,errors='coerce').fillna(0).cumsum(); return float((x-x.cummax()).min()) if len(x) else 0.0

def summarize(df, keys):
    if df.empty: return pd.DataFrame()
    rows=[]
    for key,g in df.groupby(keys,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        pnl=pd.to_numeric(g['pipeline_pnl'],errors='coerce'); ret=pd.to_numeric(g['pipeline_realized_return'],errors='coerce')
        row={k:v for k,v in zip(keys,key)}
        row.update({'trades':int(len(g)),'pnl':float(pnl.sum()),'win_rate':float((pnl>0).mean()) if len(g) else 0.0,'profit_factor':pf(pnl),'avg_return':float(ret.mean()) if len(ret.dropna()) else np.nan,'median_return':float(ret.median()) if len(ret.dropna()) else np.nan,'compound_return':float(np.prod(1+ret.dropna().to_numpy(float))-1) if len(ret.dropna()) else np.nan,'compound_max_drawdown':comp_dd(ret),'realized_pnl_drawdown':pnl_dd(pnl),'max_loss':float(pnl.min()) if len(pnl.dropna()) else 0.0})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--backtest-dir',default='portfolio_reports/backtests/historical_score_portfolio'); ap.add_argument('--saved-models',default='saved_models'); ap.add_argument('--out-dir',default=None); ap.add_argument('--initial-cash',type=float,default=None); ap.add_argument('--strict',action='store_true')
    args=ap.parse_args(); bt=Path(args.backtest_dir); bt=bt if bt.is_absolute() else PROJECT_DIR/bt; sm=Path(args.saved_models); sm=sm if sm.is_absolute() else PROJECT_DIR/sm; out=Path(args.out_dir) if args.out_dir else bt/'pipeline_consistency_audit'; out=out if out.is_absolute() else PROJECT_DIR/out; out.mkdir(parents=True,exist_ok=True)
    trades_path=bt/'historical_score_portfolio_backtest_trades.csv'
    if not trades_path.exists(): raise SystemExit(f'[ERROR] trades file not found: {trades_path}')
    trades=pd.read_csv(trades_path); sig_cache={}; sample_cache={}; rows=[]
    for _,tr in trades.iterrows():
        stock=normalize_stock_code(tr.get('stock_code')); model=norm_text(tr.get('model_name')); d=date_key(tr.get('buy_date'))
        if d not in sig_cache: sig_cache[d]=read_signals(bt,d)
        sig,status=match_signal(sig_cache[d],tr); meta_path=find_metadata(sm,stock,model,tr.get('metadata_path')); meta=read_json(meta_path)
        row=tr.to_dict(); row.update({'stock_code':stock,'model_name':model,'buy_date_key':d,'signal_match_status':status,'metadata_path_resolved':str(meta_path) if meta_path else '', 'portfolio_realized_return':as_float(tr.get('net_return'),np.nan),'portfolio_pnl':as_float(tr.get('pnl'),np.nan)})
        entry,entry_src=infer_entry(sig,meta,tr); label,label_src=infer_label(sig,meta,tr); ret_col=expected_col(label)
        target=as_float(sig.get('target_hit_bps') if sig is not None and 'target_hit_bps' in sig.index else meta.get('target_hit_bps',50.0),50.0)
        cost=as_float(sig.get('round_trip_cost_bps') if sig is not None and 'round_trip_cost_bps' in sig.index else meta.get('round_trip_cost_bps',1.7),1.7)
        prem=as_float(sig.get('entry_vwap_premium_bps') if sig is not None and 'entry_vwap_premium_bps' in sig.index else meta.get('entry_vwap_premium_bps',50.0),50.0)
        row.update({'entry_policy_checked':entry,'entry_policy_source':entry_src,'label_mode_checked':label,'label_mode_source':label_src,'expected_pipeline_return_col':ret_col,'target_hit_bps_checked':target,'round_trip_cost_bps_checked':cost,'entry_vwap_premium_bps_checked':prem})
        if sig is None: row['status']=status; rows.append(row); continue
        sp=choose_sample(sig,meta,stock); row['sample_path']=str(sp) if sp else ''
        if sp is None: row['status']='SAMPLE_PATH_MISSING'; rows.append(row); continue
        try: df=load_trade_samples(sp,sample_cache,cost,target,entry,prem)
        except Exception as exc: row['status']='ADD_TRADE_RETURNS_FAILED'; row['error']=f'{type(exc).__name__}: {exc}'; rows.append(row); continue
        if df is None or df.empty: row['status']='SAMPLE_FILE_MISSING_OR_EMPTY'; rows.append(row); continue
        hit=df[df['_date_key']==d]
        if hit.empty: row['status']='SAMPLE_ROW_MISSING'; rows.append(row); continue
        srow=hit.iloc[0]; row['entry_signal_recomputed']=boolish(srow.get('entry_signal')); row['pipeline_return_col']=ret_col
        if not ret_col: row['status']='LABEL_MODE_UNKNOWN'; rows.append(row); continue
        if ret_col not in df.columns: row['status']=f'EXPECTED_RETURN_COL_MISSING:{ret_col}'; rows.append(row); continue
        pret=as_float(srow.get(ret_col),np.nan); row['pipeline_realized_return']=pret
        if row['entry_signal_recomputed'] is not True: row['status']='ENTRY_SIGNAL_FALSE_OR_UNCOMPUTABLE'; rows.append(row); continue
        if not np.isfinite(pret): row['status']='PIPELINE_RETURN_NAN'; rows.append(row); continue
        buy_amount=as_float(tr.get('buy_amount'),np.nan)
        if not np.isfinite(buy_amount): buy_amount=as_float(tr.get('shares',tr.get('buy_shares',0.0)),0.0)*as_float(tr.get('buy_price',tr.get('price',np.nan)),np.nan)
        row['pipeline_pnl']=buy_amount*pret; row['return_diff_portfolio_minus_pipeline']=as_float(tr.get('net_return'),np.nan)-pret; row['pnl_diff_portfolio_minus_pipeline']=as_float(tr.get('pnl'),np.nan)-row['pipeline_pnl']; row['status']='OK'; rows.append(row)
    audit=pd.DataFrame(rows); audit_path=out/'portfolio_trade_pipeline_consistency_audit.csv'; audit.to_csv(audit_path,index=False,encoding='utf-8-sig')
    ok=audit[audit['status']=='OK'].copy() if 'status' in audit.columns else pd.DataFrame(); summarize(ok,['stock_code','model_name','entry_policy_checked','label_mode_checked']).to_csv(out/'pipeline_consistent_model_perf_summary.csv',index=False,encoding='utf-8-sig'); summarize(ok,['stock_code']).to_csv(out/'pipeline_consistent_stock_perf_summary.csv',index=False,encoding='utf-8-sig')
    init=args.initial_cash; js=bt/'historical_score_portfolio_backtest_summary.json'
    if init is None: init=as_float(json.loads(js.read_text(encoding='utf-8')).get('initial_cash'),0.0) if js.exists() else 0.0
    pipeline_pnl=float(pd.to_numeric(ok.get('pipeline_pnl',pd.Series(dtype=float)),errors='coerce').sum()); orig_pnl=float(pd.to_numeric(audit.get('portfolio_pnl',pd.Series(dtype=float)),errors='coerce').sum()); bad=audit[audit['status']!='OK'] if 'status' in audit.columns else audit
    summary={'status':'ok' if bad.empty else 'has_invalid_trades','trades_total':int(len(audit)),'trades_ok':int(len(ok)),'trades_invalid':int(len(bad)),'status_counts':audit['status'].value_counts(dropna=False).to_dict() if 'status' in audit.columns else {},'entry_policy_counts':audit['entry_policy_checked'].value_counts(dropna=False).to_dict() if 'entry_policy_checked' in audit.columns else {},'label_mode_counts':audit['label_mode_checked'].value_counts(dropna=False).to_dict() if 'label_mode_checked' in audit.columns else {},'pipeline_pnl_sum':pipeline_pnl,'original_portfolio_pnl_sum':orig_pnl,'pnl_diff_original_minus_pipeline':orig_pnl-pipeline_pnl,'initial_cash':init,'pipeline_consistent_final_equity_approx':float(init or 0.0)+pipeline_pnl,'audit_csv':str(audit_path),'model_perf_csv':str(out/'pipeline_consistent_model_perf_summary.csv'),'stock_perf_csv':str(out/'pipeline_consistent_stock_perf_summary.csv')}
    (out/'pipeline_consistency_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f'[AUDIT] {audit_path}')
    return 3 if args.strict and not bad.empty else 0
if __name__=='__main__': raise SystemExit(main())
