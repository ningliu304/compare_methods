"""SAM-CD 统一对比实验入口。

官方训练设置: FastSAM-x 骨干冻结，SGD(lr=0.1, momentum=0.9, nesterov,
wd=5e-4)，BCE + LatentSimilarity 损失，batch 4；本实验统一为 100 epoch。

依赖:
- ultralytics==8.0.120，并按 models/FastSAM/README.md 修改其源码
- FastSAM-x.pt（放在本目录或设 SAMCD_WEIGHTS）
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # common
sys.path.insert(0, HERE)   # models / utils

from common.config import parse_args
from common.seed import set_seed
from common.dataset import build_loaders
from common.logger import setup_logger
from common.engine import Trainer
from adapter import SAMCDAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 4, "lr": 0.1, "optim": "sgd", "nesterov": True,
        "weight_decay": 5e-4, "scheduler": "poly", "norm": "zero_one",
        "out_dir": "runs/samcd",
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), SAMCDAdapter(), logger).fit()


if __name__ == "__main__":
    main()
