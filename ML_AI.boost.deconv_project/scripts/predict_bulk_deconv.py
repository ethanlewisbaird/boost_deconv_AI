#!/usr/bin/env python3
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys

p = Path(__file__).parent.parent
cfg = yaml.safe_load((p / 'config.yaml').read_text())

sys.path.insert(0, str(p))

model_obj = joblib.load(p /'results'/'models'/'model_xgboost.joblib')
model = model_obj['model']
feature_index = list(model_obj['feature_index'])
target_columns = list(model_obj['target_columns'])

# read bulk
bulk_path = Path(cfg.get('bulk_paths',{}).get('merged'))
try:
    bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
except Exception:
    bulk = pd.read_csv(bulk_path, index_col=0)

# coerce numeric
bulk = bulk.apply(pd.to_numeric, errors='coerce').fillna(0.0)
# if duplicate gene ids exist, sum them to produce unique index
if not bulk.index.is_unique:
    bulk = bulk.groupby(bulk.index).sum()

# Align genes to features expected by model (rows = genes)
sel = feature_index
# reindex rows to the model features, filling missing genes with zeros
df_aligned = bulk.reindex(index=sel).fillna(0.0)
df_aligned.index = sel

# scale to mean_depth like pseudobulk
mean_depth = cfg.get('pseudobulk', {}).get('read_depth_mean', 1e6)
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
# predict fractions
pred = model.predict(X)
# clip and normalize per sample
pred = np.clip(pred, 0.0, None)
pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-12)

samples = list(df_aligned.columns)
res = pd.DataFrame(pred, index=samples, columns=target_columns)
out_p = p / 'results' / 'predictions_bulk_deconv.csv'
res.to_csv(out_p)
print('Wrote', out_p)
