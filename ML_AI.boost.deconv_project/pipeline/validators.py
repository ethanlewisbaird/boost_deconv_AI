import scanpy as sc
import pandas as pd
from pathlib import Path

class ValidationError(Exception):
    pass


def validate_h5ad(path):
    p = Path(path)
    if not p.exists():
        raise ValidationError(f'h5ad not found: {path}')
    try:
        ad = sc.read_h5ad(path, backed='r')
        # minimal checks
        if ad.n_obs == 0 or ad.n_vars == 0:
            raise ValidationError('Empty AnnData')
    finally:
        try:
            ad.filename = None
        except Exception:
            pass
    return True


def validate_microscopy_csv(path):
    df = pd.read_csv(path)
    req = {'Sample', 'Seurat_Cluster', 'Observed_Fraction'}
    if not req.issubset(set(df.columns)):
        raise ValidationError(f'microscopy csv missing columns: {req - set(df.columns)}')
    return True


def validate_config(cfg):
    # basic path validations (do not require data presence beyond file checks)
    if cfg.get('use_merged_sc') and 'sc_paths' in cfg:
        validate_h5ad(cfg['sc_paths']['merged'])
    if cfg.get('microscopy_csv'):
        validate_microscopy_csv(cfg['microscopy_csv'])
    return True