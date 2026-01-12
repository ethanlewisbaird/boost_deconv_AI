#!/usr/bin/env python3
"""Fast deconvolution training with reduced pseudobulk samples for quick iteration."""
import os
import yaml
from pathlib import Path
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

p = Path(__file__).resolve().parent
cfg = yaml.safe_load((p / 'config.yaml').read_text())

import sys
sys.path.insert(0, str(p))

print("Loading single-cell data...")
from pipeline.io.load_sc import load_sc_data
adata = load_sc_data(cfg['sc_paths']['merged'], is_merged=True)
print(f"Loaded: {adata.shape}")

print("Generating fractional pseudobulk (100 samples for speed)...")
cfg_fast = dict(cfg)
cfg_fast['pseudo_bulk_samples'] = 100  # Fast mode
from pipeline.pseudobulk.advanced_pseudobulk import generate_fractional_pseudobulk
pb = generate_fractional_pseudobulk(adata, cfg_fast)
print('Generated fractional pseudobulk:', pb['pseudobulk'].shape)

print("Training deconvolution model...")
X_df = pb['pseudobulk'].T  # samples x genes
meta = pb['meta']
y_cols = [c for c in meta.columns if c.startswith('frac_')]
if len(y_cols) == 0:
    raise RuntimeError('No fractional columns found')
Y_df = meta[y_cols].astype(float)

X = X_df.values
Y = Y_df.values

base = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
mor = MultiOutputRegressor(base, n_jobs=1)
mor.fit(X, Y)

os.makedirs(os.path.join(str(p), 'results', 'models'), exist_ok=True)
model_path = p / 'results' / 'models' / 'model_deconv.joblib'
joblib.dump({'model': mor, 'feature_index': list(X_df.columns), 'target_columns': y_cols}, model_path)
print('Saved deconv model to', model_path)
