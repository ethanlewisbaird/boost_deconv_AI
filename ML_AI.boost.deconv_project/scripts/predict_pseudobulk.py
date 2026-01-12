import yaml
from pathlib import Path
import joblib
import pandas as pd

# project root
p = Path(__file__).parent.parent
cfg = yaml.safe_load((p / 'config.yaml').read_text())

# ensure venv python has access to project package
import sys
sys.path.insert(0, str(p))

from pipeline.io.load_sc import load_sc_data
from pipeline.pseudobulk.advanced_pseudobulk import generate_advanced_pseudobulk

# load model
model_path = p / 'results' / 'models' / 'model_xgboost.joblib'
if not model_path.exists():
    raise SystemExit(f'Model not found: {model_path}')
model = joblib.load(model_path)

# regenerate pseudobulk (same config)
adata = load_sc_data(cfg['sc_paths']['merged'], is_merged=True)
pb = generate_advanced_pseudobulk(adata, cfg)

genes = list(pb['pseudobulk'].index)
samples = pb['pseudobulk'].columns.tolist()

# Align features with model expectation
n_model = getattr(model, 'n_features_in_', None)
if n_model is not None and len(genes) != n_model:
    # try to get stored feature names from model
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
        feature_names = feature_names or None

    if feature_names:
        # intersect in model order
        sel = [f for f in feature_names if f in genes]
        if len(sel) >= n_model:
            sel = sel[:n_model]
            X = pb['pseudobulk'].loc[sel].T.values
        else:
            # not enough overlap; fallback to trimming
            X = pb['pseudobulk'].T.values[:, :n_model]
    else:
        # No feature names stored; deterministically trim to first n_model genes
        X = pb['pseudobulk'].T.values[:, :n_model]
else:
    X = pb['pseudobulk'].T.values
classes = None

# get probabilities or predictions
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)
    # sklearn predict_proba returns columns in order of model.classes_
    classes = list(model.classes_)
    dfp = pd.DataFrame(probs, index=samples, columns=[str(c) for c in classes])
    dfp.index.name = 'sample'
    out_p = p / 'results' / 'predictions_pseudobulk_proba.csv'
    dfp.to_csv(out_p)
else:
    preds = model.predict(X)
    dfp = pd.DataFrame({'predicted': preds}, index=samples)
    dfp.index.name = 'sample'
    out_p = p / 'results' / 'predictions_pseudobulk_pred.csv'
    dfp.to_csv(out_p)

# also save predicted labels
preds = model.predict(X)
dfl = pd.DataFrame({'predicted_label': preds}, index=samples)
dfl.index.name = 'sample'
out_l = p / 'results' / 'predicted_labels_pseudobulk.csv'
dfl.to_csv(out_l)

print('Wrote:', out_p, out_l)
