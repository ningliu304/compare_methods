"""ChangeMamba 统一对比实验入口。

用法（从根目录）:
    python ChangeMamba-master/train_cd.py --root data --out-dir runs/changemamba

可选：设 CHANGEMAMBA_PRETRAINED=/path/to/vssm_tiny_ckpt.pth 加载 VMamba 预训练编码器。

依赖: 已编译的 SSM 算子（cd ChangeMamba-master/kernels/selective_scan && pip install .）。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # common
sys.path.insert(0, HERE)   # changedetection/

from common.config import parse_args
from common.seed import set_seed
from common.dataset import build_loaders
from common.logger import setup_logger
from common.engine import Trainer
from adapter import ChangeMambaAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 4, "lr": 1e-4, "optim": "adamw", "weight_decay": 5e-3,
        "scheduler": "cosine", "norm": "imagenet", "out_dir": "runs/changemamba",
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), ChangeMambaAdapter(), logger).fit()


if __name__ == "__main__":
    main()
