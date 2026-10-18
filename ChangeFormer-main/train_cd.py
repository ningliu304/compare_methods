"""ChangeFormer 统一对比实验入口。

用法:
    python train_cd.py --root ../data --out-dir runs/changeformer
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
from adapter import ChangeFormerAdapter


def main():
    cfg = parse_args(defaults={
        # 官方设置 (LEVIR-CD): AdamW(lr=1e-4, wd=0.01, betas=(0.9,0.999)) + linear decay
        # 官方 batch=16 / 200 epoch / 256 输入；本实验统一为 512 输入、100 epoch，batch 保持 8
        "batch_size": 8, "lr": 1e-4, "optim": "adamw",
        "weight_decay": 0.01, "scheduler": "linear", "out_dir": "runs/changeformer",
        "norm": "minus_one_one",   # 与原作者实现一致
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), ChangeFormerAdapter(), logger).fit()


if __name__ == "__main__":
    main()
