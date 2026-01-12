import logging
import os
from pathlib import Path

def setup_logging(log_file: str = 'results/logs/pipeline.log', level: str = 'INFO'):
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('ml_boost_deconv')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger