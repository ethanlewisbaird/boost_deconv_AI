import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys

p = Path(__file__).parent.parent
cfg = yaml.safe_load((p / 'config.yaml').read_text())

# ensure imports find project
sys.path.insert(0, str(p))
from pipeline.io.load_sc import load_sc_data

# load model
model_path = p / 'results' / 'models' / 'model_xgboost.joblib'
if not model_path.exists():
    raise SystemExit(f'Model not found: {model_path}')
model = joblib.load(model_path)

# locate bulk merged path in config
bulk_path = cfg.get('bulk_paths', {}).get('merged')
if not bulk_path:
    raise SystemExit('bulk_paths.merged not set in config.yaml')
bulk_path = Path(bulk_path)
if not bulk_path.exists():
    raise SystemExit(f'Bulk file not found: {bulk_path}')

# read bulk counts; try tab/tsv or csv
try:
    bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
except Exception:
    bulk = pd.read_csv(bulk_path, index_col=0)

# ensure genes are the index (rows)
# if genes are columns (many cols with non-numeric gene names), transpose heuristics
if bulk.shape[0] < bulk.shape[1] and bulk.index.dtype == 'int64':
    # unlikely but handle
    bulk = bulk.T

# Align genes to model expectation
genes = list(bulk.index)
n_model = getattr(model, 'n_features_in_', None)
feature_names = None
try:
    feature_names = list(getattr(model, 'feature_names_in_', None) or [])
except Exception:
    feature_names = None
try:
    if not feature_names:
        booster = model.get_booster()
        feature_names = list(booster.feature_names) if booster.feature_names is not None else None
except Exception:
    pass

# Build X matrix aligned to model features
if n_model is not None:
    if feature_names:
        # intersection in model order; if some model features missing in bulk, fill zeros
        sel = feature_names
        # create DataFrame with those features, filling missing with zeros
        df_aligned = pd.DataFrame(index=sel)
        bulk_series = bulk
        # for each feature, if present in bulk use it otherwise zeros
        cols = []
        for f in sel:
            if f in bulk.index:
                cols.append(bulk.loc[f].astype(float))
            else:
                cols.append(pd.Series(0.0, index=bulk.columns))
        df_aligned = pd.concat(cols, axis=1)
        df_aligned.columns = sel
        # ensure ordering and shape
        X_raw = df_aligned.T.values  # features x samples
    else:
        # no feature names, deterministically choose first n_model genes from bulk index
        if len(genes) < n_model:
            # pad with zeros
            pad = n_model - len(genes)
            df2 = bulk.copy()
            for i in range(pad):
                df2[f'pad_gene_{i}'] = 0.0
            X_raw = df2.values
        else:
            X_raw = bulk.iloc[:n_model].values
else:
    # model has no n_features_in_ (unlikely) — use all genes present in both
    common = [g for g in genes if g in getattr(model, 'feature_names_in_', genes)]
    if len(common) == 0:
        X_raw = bulk.values
    else:
        X_raw = bulk.loc[common].values

# X_raw is features x samples; we need samples x features for model
# but we will apply scaling per-sample first
# Convert to DataFrame for convenience
if 'df_aligned' in locals():
    df_features = df_aligned
else:
    # reconstruct DataFrame from X_raw
    samples = list(bulk.columns)
    if X_raw.shape[1] == len(samples):
        df_features = pd.DataFrame(X_raw, index=(feature_names if feature_names else list(bulk.index)[:X_raw.shape[0]]), columns=samples)
    else:
        # transpose fallback
        df_features = pd.DataFrame(X_raw, index=(feature_names if feature_names else list(bulk.index)[:X_raw.shape[0]]))

# Now scale each sample to mean_depth used in pseudobulk
mean_depth = cfg.get('pseudobulk', {}).get('read_depth_mean', 1e6)
# For each sample, get counts vector, compute proportions, scale to mean_depth
scaled = []
for s in df_features.columns:
    arr = df_features[s].astype(float).values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    tot = arr.sum()
    if tot <= 0 or not np.isfinite(tot):
        scaled_arr = np.zeros_like(arr, dtype=int)
    else:
        prop = arr / tot
        scaled_arr = np.round(prop * float(mean_depth)).astype(int)
    scaled.append(scaled_arr)

# Create X matrix samples x features
X = np.vstack(scaled)
samples = list(df_features.columns)

# If model expects fewer features, trim columns; if more, pad zeros
if n_model is not None:
    if X.shape[1] > n_model:
        X = X[:, :n_model]
    elif X.shape[1] < n_model:
        pad = np.zeros((X.shape[0], n_model - X.shape[1]), dtype=int)
        X = np.concatenate([X, pad], axis=1)

# Run predictions
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    dfp = pd.DataFrame(probs, index=samples, columns=[str(c) for c in classes])
    dfp.index.name = 'sample'
    out_p = p / 'results' / 'predictions_bulk_proba.csv'
    dfp.to_csv(out_p)
else:
    preds = model.predict(X)
    dfp = pd.DataFrame({'predicted': preds}, index=samples)
    dfp.index.name = 'sample'
    out_p = p / 'results' / 'predictions_bulk_pred.csv'
    dfp.to_csv(out_p)

# save labels
preds = model.predict(X)
dfl = pd.DataFrame({'predicted_label': preds}, index=samples)
dfl.index.name = 'sample'
out_l = p / 'results' / 'predicted_labels_bulk.csv'
dfl.to_csv(out_l)

print('Wrote:', out_p, out_l)
