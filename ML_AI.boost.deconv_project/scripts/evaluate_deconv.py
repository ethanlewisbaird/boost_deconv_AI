#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

p = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((p / 'config.yaml').read_text())

# local imports
import sys
sys.path.insert(0, str(p))
from pipeline.io.load_sc import load_sc_data
from pipeline.pseudobulk.advanced_pseudobulk import generate_fractional_pseudobulk

MODES = [
    {'name': 'none', 'use_microscopy_augmentation': False},
    {'name': 'global', 'use_microscopy_augmentation': True, 'microscopy_mode': 'global'},
    {'name': 'per_sample', 'use_microscopy_augmentation': True, 'microscopy_mode': 'per_sample'},
]

results = {}
adata = load_sc_data(cfg['sc_paths']['merged'], is_merged=True)

for m in MODES:
    name = m['name']
    outdir = p / 'results' / f'eval_{name}'
    outdir.mkdir(parents=True, exist_ok=True)
    print('\n=== Running mode:', name)
    # prepare local config copy
    cfg_local = dict(cfg)
    cfg_local['use_microscopy_augmentation'] = m.get('use_microscopy_augmentation', False)
    if 'microscopy_mode' in m:
        cfg_local['microscopy_mode'] = m['microscopy_mode']
    cfg_local['output_dir'] = str(outdir)

    pb = generate_fractional_pseudobulk(adata, cfg_local)
    X_df = pb['pseudobulk'].T  # samples x genes
    meta = pb['meta']
    y_cols = [c for c in meta.columns if c.startswith('frac_')]
    if len(y_cols) == 0:
        raise RuntimeError('No fractional columns found in pseudo_bulk.meta')
    Y_df = meta[y_cols].astype(float)

    X = X_df.values
    Y = Y_df.values

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    base = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    mor = MultiOutputRegressor(base, n_jobs=1)
    mor.fit(X_train, y_train)

    y_pred = mor.predict(X_test)
    # ensure non-negative and normalize per sample for fair RMSE
    y_pred = np.clip(y_pred, 0.0, None)
    y_pred = y_pred / (y_pred.sum(axis=1, keepdims=True) + 1e-12)

    # compute per-target RMSE
    per_target_rmse = np.sqrt(np.mean((y_test - y_pred)**2, axis=0))
    overall_rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    metrics = {
        'mode': name,
        'overall_rmse': float(overall_rmse),
        'per_target_rmse': {y_cols[i]: float(per_target_rmse[i]) for i in range(len(y_cols))},
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
    }

    # save metrics
    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # save model (trained on full data for deployment)
    base_full = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    mor_full = MultiOutputRegressor(base_full, n_jobs=1)
    mor_full.fit(X, Y)
    model_obj = {'model': mor_full, 'feature_index': list(X_df.columns), 'target_columns': y_cols}
    (outdir / 'models').mkdir(exist_ok=True)
    joblib.dump(model_obj, outdir / 'models' / 'model_deconv.joblib')

    # produce bulk predictions using this model
    # read bulk
    bulk_path = Path(cfg_local.get('bulk_paths',{}).get('merged'))
    try:
        bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
    except Exception:
        bulk = pd.read_csv(bulk_path, index_col=0)
    bulk = bulk.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if not bulk.index.is_unique:
        bulk = bulk.groupby(bulk.index).sum()

    sel = model_obj['feature_index']
    df_aligned = bulk.reindex(index=sel).fillna(0.0)
    mean_depth = cfg_local.get('pseudobulk', {}).get('read_depth_mean', 1e6)
    scaled = []
    for s in df_aligned.columns:
        arr = df_aligned[s].astype(float).values
        tot = arr.sum()
        if tot <= 0 or not np.isfinite(tot):
            scaled_arr = np.zeros_like(arr, dtype=float)
        else:
            prop = arr / tot
            scaled_arr = prop * float(mean_depth)
        scaled.append(scaled_arr)
    X_bulk = np.vstack(scaled)
    pred = mor_full.predict(X_bulk)
    pred = np.clip(pred, 0.0, None)
    pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-12)
    samples = list(df_aligned.columns)
    res = pd.DataFrame(pred, index=samples, columns=y_cols)
    res.to_csv(outdir / 'predictions_bulk_deconv.csv')

    results[name] = metrics
    print('Mode', name, 'done. Overall RMSE:', overall_rmse)

# write aggregate results
(p / 'results' / 'eval_summary.json').write_text(json.dumps(results, indent=2))
print('\nAll evaluations complete. Summary written to results/eval_summary.json')
