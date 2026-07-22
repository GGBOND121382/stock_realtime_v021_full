#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from utils import as1455_ch17_common as common

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-data',required=True); p.add_argument('--feature-presets',default='rotation_onehot rotation_addon_onehot'); p.add_argument('--targets',default='r01_fwd r05_fwd r21_fwd'); a=p.parse_args()
    rows=[]
    for target in a.targets.split():
        for preset in a.feature_presets.split():
            d=common.fold_dir_from_template(common.default_fold_dir_template(preset,target),0)
            found=None
            for path in [d/'fold_report.json', d/'preprocess'/'feature_manifest.json']:
                if path.is_file():
                    o=json.loads(path.read_text(encoding='utf-8')); raw=o.get('validation_end') or o.get('test_end') or o.get('fold_end')
                    if raw: found=(pd.Timestamp(raw).normalize(),path); break
            if found is None: raise FileNotFoundError(f'fold0 boundary not found: {d}')
            rows.append((target,preset,found[0],str(found[1])))
    boundary=max(x[2] for x in rows)
    data=pd.read_hdf(a.model_data,'model_data'); outcomes=list(common.base.EXPECTED_OUTCOMES); feature_cols=[c for c in data.columns if c not in outcomes]
    valid=data.dropna(subset=feature_cols); dates=pd.DatetimeIndex(valid.index.get_level_values('date')).normalize(); future=dates[dates>boundary]
    if len(future)==0: raise RuntimeError(f'no feature-complete date after {boundary:%Y-%m-%d}')
    start=pd.Timestamp(future.min()).normalize()
    print(json.dumps({'common_fold0_end':boundary.strftime('%Y-%m-%d'),'common_forward_start':start.strftime('%Y-%m-%d'),'fold0_boundaries':[{'target':t,'preset':p,'fold0_end':e.strftime('%Y-%m-%d'),'source':s} for t,p,e,s in rows]},ensure_ascii=False),file=sys.stderr)
    print(start.strftime('%Y-%m-%d'))
if __name__=='__main__': main()
