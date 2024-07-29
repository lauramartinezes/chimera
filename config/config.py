from configparser import ConfigParser
from pathlib import Path

# Directories of interest
cfg = ConfigParser()
cfg.read('config/config.ini')
working_at = str(cfg.get('base', 'working_at'))
DATA_DIR = Path(cfg.get(working_at, 'data_dir'))
REPO_DIR = Path(cfg.get(working_at, 'repo_dir'))
SAVE_DIR = Path(cfg.get(working_at, 'save_dir'))
RESULTS_DIR = Path(cfg.get(working_at, 'results_dir'))

INSECT_LABELS_MAP = {'bl':0,'wswl':1,'sp':2,'t':3,'sw':4,'k':5,'m':6,'c':7,'v':8,'wmv':9,'wrl':10,'other':11}
WANDB_PROJECT = cfg.get(working_at, 'wandb_project')
WANDB_ENTITY = cfg.get(working_at, 'wandb_entity')