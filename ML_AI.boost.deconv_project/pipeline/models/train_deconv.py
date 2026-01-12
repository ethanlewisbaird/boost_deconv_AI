import os
import yaml
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_deconv(pseudo_bulk: dict, config: dict, outdir='results'):
    """Train a multi-output regressor to predict cluster fractions from pseudobulk.

    pseudo_bulk: dict with 'pseudobulk' (genes x samples DataFrame) and 'meta' (samples x frac_* columns)
    """
    X_df = pseudo_bulk['pseudobulk'].T  # samples x genes
    meta = pseudo_bulk['meta']
    # select fraction columns (cols starting with 'frac_')
    y_cols = [c for c in meta.columns if c.startswith('frac_')]
    if len(y_cols) == 0:
        raise RuntimeError('No fractional columns found in pseudo_bulk.meta; expected columns starting with frac_')
    Y_df = meta[y_cols].astype(float)

    X = X_df.values
    Y = Y_df.values

    # instantiate estimator
    base = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
    mor = MultiOutputRegressor(base, n_jobs=1)

    mor.fit(X, Y)

    os.makedirs(os.path.join(outdir, 'models'), exist_ok=True)
    model_path = os.path.join(outdir, 'models', 'model_deconv.joblib')
    joblib.dump({'model': mor, 'feature_index': list(X_df.columns), 'target_columns': y_cols}, model_path)
    return model_path


if __name__ == '__main__':
    from pathlib import Path
    p = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(p / 'config.yaml'))
    # load sc and generate fractional pseudobulk
    from pipeline.io.load_sc import load_sc_data
    from pipeline.pseudobulk.advanced_pseudobulk import generate_fractional_pseudobulk

    adata = load_sc_data(cfg['sc_paths']['merged'], is_merged=True)
    pb = generate_fractional_pseudobulk(adata, cfg)
    print('Generated fractional pseudobulk:', pb['pseudobulk'].shape)
    model_path = train_deconv(pb, cfg, outdir=cfg.get('output_dir','results'))
    print('Saved deconv model to', model_path)
