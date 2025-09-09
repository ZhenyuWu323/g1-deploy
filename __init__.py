import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()



CONFIG_DIR = PROJECT_ROOT / 'configs'
CHEKPOINT_DIR = PROJECT_ROOT / 'policy' / 'checkpoints'


for directory in [CONFIG_DIR, CHEKPOINT_DIR]:
    directory.mkdir(exist_ok=True)


__all__ = ['PROJECT_ROOT', 'CONFIG_DIR', 'CHEKPOINT_DIR']

