"""TinyCD 统一对比实验入口。

用法:
    python train_cd.py --root ../data --out-dir runs/tinycd
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from common.config import parse_args
from common.seed import set_seed
from common.dataset import build_loaders
from common.logger import setup_logger
from common.engine import Trainer
from adapter import TinyCDAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 8,
        "lr": 0.00356799066427741,
        "weight_decay": 0.009449677083344786,
        "optim": "adamw", "scheduler": "cosine", "out_dir": "runs/tinycd",
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), TinyCDAdapter(), logger).fit()


if __name__ == "__main__":
    main()
