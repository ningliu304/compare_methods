"""ChangeViT 统一对比实验入口。

用法（从 ChangeViT-master 目录运行，便于仓库内相对权重路径生效）:
    python train_cd.py --root ../data --out-dir ../runs/changevit
或从根目录:
    python ChangeViT-master/train_cd.py --root data --out-dir runs/changevit

依赖: torch、torchvision、einops；预训练权重放 checkpoint/ 下。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # common
sys.path.insert(0, HERE)   # model/

from common.config import parse_args
from common.seed import set_seed
from common.dataset import build_loaders
from common.logger import setup_logger
from common.engine import Trainer
from adapter import ChangeViTAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 8, "lr": 2e-4, "optim": "adam", "weight_decay": 1e-4,
        "scheduler": "poly", "norm": "imagenet", "out_dir": "runs/changevit",
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), ChangeViTAdapter(model_type="small"), logger).fit()


if __name__ == "__main__":
    main()
