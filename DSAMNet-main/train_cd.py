"""DSAMNet 统一对比实验入口。

用法:
    python train_cd.py --root ../data --out-dir runs/dsamnet
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
from adapter import DSAMNetAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 8, "lr": 5e-4, "optim": "adam",
        "scheduler": "none", "out_dir": "runs/dsamnet",
        "norm": "minus_one_one",   # 与原 DSAMNet 实现一致
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), DSAMNetAdapter(), logger).fit()


if __name__ == "__main__":
    main()
