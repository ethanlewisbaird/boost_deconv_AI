import os
from pathlib import Path
import json
from .logging_config import setup_logging
from .validators import validate_config
from .io.load_sc import load_sc_data
_LAZY_IMPORTS = True

# Defer heavy/optional imports until runtime so the package can be imported
# even when some submodules or optional components are missing.


def run_pipeline(config: dict):
    logger = setup_logging(config.get('logging',{}).get('file','results/logs/pipeline.log'), config.get('logging',{}).get('level','INFO'))
    logger.info('Starting pipeline v7')
    outdir = Path(config.get('output_dir','results'))
    outdir.mkdir(parents=True, exist_ok=True)
    # save run metadata
    run_meta = {'config': config}
    with open(outdir / 'run_metadata.json','w') as f:
        json.dump(run_meta, f, indent=2)
    # validate
    try:
        validate_config(config)
        logger.info('Config validation passed')
    except Exception as e:
        logger.error('Validation failed: %s', e)
        raise
    # load sc
    if config.get('use_merged_sc', True):
        adata = load_sc_data(config['sc_paths']['merged'], is_merged=True)
    else:
        adata = load_sc_data(config['sc_paths']['libraries'], is_merged=False)
    # filter
    if config.get('remove_mt_rpl', True):
        from .preprocessing.filter_genes import filter_mt_rpl
        adata = filter_mt_rpl(adata)
    # generate pseudobulk (import lazily so missing modules don't prevent import)
    try:
        from .pseudobulk.advanced_pseudobulk import generate_advanced_pseudobulk
    except Exception:
        raise RuntimeError('pseudobulk module not available; cannot continue')
    pb = generate_advanced_pseudobulk(adata, config)

    # train (lazy import)
    try:
        from .models.train_v2 import train_model
        from .models.evaluate import evaluate_model
    except Exception:
        raise RuntimeError('models module not available; cannot continue')

    model = train_model(pb, config)
    # evaluate
    eval_res = evaluate_model(model, pb, config)
    # calibration example (if predictions & truth available)
    logger.info('Pipeline finished')
    print('Results in', outdir)
