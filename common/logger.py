"""统一日志：同时输出到控制台与 <out_dir>/log.txt，格式跨模型一致。"""

import logging
import os
import sys


def setup_logger(out_dir: str, filename: str = "log.txt", name: str = "cd") -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 避免重复添加 handler（多次调用）
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(out_dir, filename), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


EPOCH_HEADER = "epoch | train_loss | val_loss | Precision | Recall | F1 | IoU"
METRIC_ROW = "{epoch:>5} | {tl:>10.4f} | {vl:>8.4f} | {p:>9.4f} | {r:>6.4f} | {f1:>6.4f} | {iou:>6.4f}"


def format_epoch_row(epoch, train_loss, val_stats, val_loss=None):
    return METRIC_ROW.format(
        epoch=epoch,
        tl=train_loss,
        vl=(val_loss if val_loss is not None else float("nan")),
        p=val_stats["precision"],
        r=val_stats["recall"],
        f1=val_stats["f1"],
        iou=val_stats["iou"],
    )
