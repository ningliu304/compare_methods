"""MMSeg 系模型的统一落盘工具。

训练 Hook 与评测脚本共用，保证 CMCDNet/ACTNet/FAEWNet 的
best_<数据集>.pth、log_<数据集>.txt、vis_<数据集>/ 与统一 Trainer 一致。
"""

import datetime
import os

from common.logger import EPOCH_HEADER, METRIC_ROW


def dataset_suffix(dataset: str) -> str:
    return f"_{dataset}" if dataset else ""


def resolve_dataset(explicit: str, data_root: str) -> str:
    if explicit:
        return explicit
    return os.path.basename(os.path.normpath(data_root)) or ""


def artifact_paths(out_dir: str, dataset: str):
    suffix = dataset_suffix(dataset)
    return (
        os.path.join(out_dir, f"best{suffix}.pth"),
        os.path.join(out_dir, f"log{suffix}.txt"),
        os.path.join(out_dir, f"vis{suffix}"),
    )


def to_float(value, default=None):
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_lines(log_path: str, lines):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{stamp} | {line}\n")


def append_epoch_header(log_path):
    _append_lines(log_path, [EPOCH_HEADER])


def append_epoch_row(log_path, epoch, train_loss, val_loss, precision,
                     recall, f1, iou):
    new_file = not os.path.isfile(log_path) or os.path.getsize(log_path) == 0
    lines = [EPOCH_HEADER] if new_file else []
    lines.append(METRIC_ROW.format(
        epoch=epoch,
        tl=train_loss,
        vl=(val_loss if val_loss is not None else float("nan")),
        p=precision,
        r=recall,
        f1=f1,
        iou=iou,
    ))
    _append_lines(log_path, lines)


def append_test_result(log_path, title, res, vis_dir=None):
    lines = [
        "=" * 60,
        f"[TEST] {title}",
        "  Precision={precision:.4f}  Recall={recall:.4f}  "
        "F1={f1:.4f}  IoU={iou:.4f}".format(**res),
        "  mIoU={miou:.4f}  mF1={mf1:.4f}  OA={acc:.4f}".format(**res),
    ]
    if vis_dir:
        lines.append(f"  visualizations -> {vis_dir}")
    lines.append("=" * 60)
    _append_lines(log_path, lines)
