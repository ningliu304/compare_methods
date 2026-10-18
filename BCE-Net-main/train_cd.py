"""BCE-Net（双时相二值 CD 改造版）统一对比实验入口。

用法:
    python train_cd.py --root ../data --out-dir runs/bcenet

说明:
- 使用 model_bcd.py::Baseline34BCD（BCE-Net 编码器几何 + 双时相差分 U 形解码器）。
- 若本机已编译 DCNv2 则使用原可变形卷积，否则自动退化为普通卷积。
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
from adapter import BCENetAdapter


def main():
    cfg = parse_args(defaults={
        "batch_size": 4, "lr": 1e-4, "optim": "adam",
        "scheduler": "cosine", "out_dir": "runs/bcenet",
        "norm": "zero_one",   # 与 BCE-Net 原实现一致（Normalize([0],[1])）
    })
    set_seed(cfg.seed)
    logger = setup_logger(cfg.out_dir, cfg.log_name)
    logger.info(f"cfg = {cfg}")
    Trainer(cfg, build_loaders(cfg), BCENetAdapter(), logger).fit()


if __name__ == "__main__":
    main()
