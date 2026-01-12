import yaml
from pathlib import Path

p = Path(__file__).parent
cfg_path = p / 'config.yaml'
if not cfg_path.exists():
    raise SystemExit('config.yaml not found in project root')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

from pipeline import main

if __name__ == '__main__':
    main.run_pipeline(cfg)
