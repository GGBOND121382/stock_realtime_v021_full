#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
PROJECT_DIR=Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0,str(PROJECT_DIR))
from utils.as1455_model_selection import find_summary_file,read_csv_auto,select_best_run
from utils.as1455_plotting import plot_frequency
RULES={'daily':None,'weekly':'W-FRI','monthly':'M'}

def args():
 p=argparse.ArgumentParser(); p.add_argument('--historical-root',action='append',required=True); p.add_argument('--forward-root',action='append',required=True); p.add_argument('--label',action='append',required=True); p.add_argument('--rank-metric',default='sharpe'); p.add_argument('--frequencies',default='daily,weekly,monthly'); p.add_argument('--out-dir',required=True); return p.parse_args()
def nav_file(root,grid,run):
 for p in [grid/'01_runs'/run/'close_auction_nav.csv',root/'01_close_auction_grid'/'01_runs'/run/'close_auction_nav.csv',root/'01_close_auction_daily_grid'/'01_runs'/run/'close_auction_nav.csv']:
  if p.is_file(): return p
 m=sorted(root.glob(f'**/01_runs/{run}/close_auction_nav.csv'))
 if m:return m[0]
 raise FileNotFoundError(f'NAV not found: {root} {run}')
def load_nav(p):
 d=read_csv_auto(p)
 if not {'date','nav'}<=set(d.columns): raise RuntimeError(f'{p} lacks date/nav')
 d['date']=pd.to_datetime(d['date'],errors='coerce').dt.normalize(); d['nav']=pd.to_numeric(d['nav'],errors='coerce')
 if 'nav_before_trade' in d:d['nav_before_trade']=pd.to_numeric(d['nav_before_trade'],errors='coerce')
 return d.dropna(subset=['date','nav']).sort_values('date').drop_duplicates('date',keep='last')
def select(root,metric):
 s,g=find_summary_file(root); b=select_best_run(read_csv_auto(s),metric); run=str(b['run_name']); p=nav_file(root,g,run); return load_nav(p),{'root':str(root),'run_name':run,'nav_file':str(p)}
def mapping(root):
 p=root/'00_predictions'/'one_lag_prediction_manifest.json'; o=json.loads(p.read_text(encoding='utf-8')); out={}
 for r in o.get('fold_mapping',[]):
  sf=int(r['source_fold']); st=r.get('target_fold_start') or r.get('target_validation_start') or r.get('target_test_start'); en=r.get('target_fold_end') or r.get('target_validation_end') or r.get('target_test_end')
  if st and en: out[sf]={**r,'start':st,'end':en}
 if not out: raise RuntimeError(f'no fold mappings: {p}')
 return out
def norm(d,st,en):
 x=d[(d.date>=st)&(d.date<=en)].copy()
 if x.empty: raise RuntimeError(f'empty NAV {st}..{en}')
 base=None
 if 'nav_before_trade' in x:
  v=x.iloc[0]['nav_before_trade']; base=float(v) if pd.notna(v) and float(v)>0 else None
 if base is None: base=float(x.iloc[0].nav)
 x['return_pct']=(x.nav.astype(float)/base-1)*100
 return x[['date','nav','return_pct']]
def sample(d,f):
 if f=='daily':return d.copy()
 return d.set_index('date')[['nav','return_pct']].resample(RULES[f]).last().dropna().reset_index()
def main():
 a=args()
 if not(len(a.historical_root)==len(a.forward_root)==len(a.label)):raise SystemExit('root/label counts differ')
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); ss=[]; selected=[]
 for h,f,l in zip(a.historical_root,a.forward_root,a.label):
  hn,hm=select(Path(h),a.rank_metric); fn,fm=select(Path(f),a.rank_metric); ss.append({'label':l,'h':hn,'f':fn,'map':mapping(Path(h))}); selected.append({'label':l,'historical':hm,'forward':fm})
 manifest={'strategies':selected,'folds':{}}
 for fold in range(6,-1,-1):
  av=[]; starts=[]; ends=[]
  for s in ss:
   if fold==0: d=s['f']; st=pd.Timestamp(d.date.min()).normalize(); en=pd.Timestamp(d.date.max()).normalize()
   else:
    m=s['map'].get(fold)
    if m is None: continue
    d=s['h']; st=pd.Timestamp(m['start']).normalize(); en=pd.Timestamp(m['end']).normalize()
   av.append((s['label'],d)); starts.append(st); ends.append(en)
  if not av: raise RuntimeError(f'fold{fold} has no strategy')
  st=max(starts); en=min(ends)
  if st>en: raise RuntimeError(f'fold{fold} has no common interval')
  curves=[{'label':l,'run_name':f'fold{fold}','curve':norm(d,st,en)} for l,d in av]; fd=out/f'fold{fold}'; fd.mkdir(parents=True,exist_ok=True)
  manifest['folds'][f'fold{fold}']={'common_start':st.strftime('%Y-%m-%d'),'common_end':en.strftime('%Y-%m-%d'),'strategies':[l for l,_ in av]}
  for freq in [x.strip() for x in a.frequencies.split(',') if x.strip()]:
   frame=plot_frequency(curves=curves,frequency=freq,out_file=fd/f'return_curve_{freq}.png',title=f'AS1455 fold{fold} return ({freq})',sample_curve=sample,plt=plt); frame.to_csv(fd/f'return_curve_{freq}.csv',index=False,encoding='utf-8-sig'); print(f'[OK] fold{fold} {freq}')
 pd.DataFrame([{'label':x['label'],'historical_root':x['historical']['root'],'historical_run':x['historical']['run_name'],'forward_root':x['forward']['root'],'forward_run':x['forward']['run_name']} for x in selected]).to_csv(out/'selected_runs.csv',index=False,encoding='utf-8-sig')
 (out/'fold_sequence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'[DONE] {out}')
if __name__=='__main__':main()
