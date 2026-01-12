#!/usr/bin/env python3
"""Fast bulk prediction using the trained deconvolution model."""
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys

p = Path(__file__).resolve().parent
cfg = yaml.safe_load((p / 'config.yaml').read_text())

sys.path.insert(0, str(p))

print("Loading deconvolution model...")
model_obj = joblib.load(p / 'results' / 'models' / 'model_deconv.joblib')
model = model_obj['model']
feature_index = list(model_obj['feature_index'])
target_columns = list(model_obj['target_columns'])

print(f"Model features: {len(feature_index)}")
print(f"Target columns (clusters): {target_columns}")

# read bulk
print("Loading bulk data...")
bulk_path = Path(cfg.get('bulk_paths', {}).get('merged'))
try:
    bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
except Exception:
    bulk = pd.read_csv(bulk_path, index_col=0)

bulk = bulk.apply(pd.to_numeric, errors='coerce').fillna(0.0)
if not bulk.index.is_unique:
    bulk = bulk.groupby(bulk.index).sum()

print(f"Bulk data shape: {bulk.shape}")

# Align genes to model features
df_aligned = bulk.reindex(index=feature_index).fillna(0.0)
mean_depth = cfg.get('pseudobulk', {}).get('read_depth_mean', 1e6)

print("Scaling bulk to pseudobulk read depth...")
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

X = np.vstack(scaled)

print("Predicting fractions...")
pred = model.predict(X)

# Ensure non-negative and normalize
pred = np.clip(pred, 0.0, None)
pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-12)

samples = list(df_aligned.columns)
res = pd.DataFrame(pred, index=samples, columns=target_columns)

out_p = p / 'results' / 'predictions_bulk_deconv.csv'
res.to_csv(out_p)
print(f'Wrote predictions to {out_p}')
print(res.head())
