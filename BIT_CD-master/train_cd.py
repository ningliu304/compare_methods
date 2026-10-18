"""BIT_CD 统一对比实验入口。

用法:
    python train_cd.py --root ../data --out-dir runs/bit_cd
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # 使 `common` 可导入
sys.path.insert(0, HERE)   # 使仓库内 `models` 等包可导入

from common.config import parse_args
from common.seed import set_seed
from common.dataset import build_loaders
from common.logger import setup_logger
from common.engine import Trainer
from adapter import BITAdapter


def main():
    cfg = parse_args(defaults={
        # 官方设置: SGD(lr=0.01, momentum=0.9, wd=5e-4) + linear decay
        "batch_size": 8, "lr": 0.01, "optim": "sgd",
        "weight_decay": 5e-4, "scheduler": "linear", "out_dir": "runs/bit_cd",
        "norm": "minus_one_one",   # 与原 BIT 实现一致；如需严格统一可 --norm imagenet
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), BITAdapter(), logger).fit()


if __name__ == "__main__":
    main()
