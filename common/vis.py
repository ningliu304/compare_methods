"""推理结果可视化：保存 A | B | GT | Pred 四联对比图。"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from common.config import resolve_norm


def _denorm(img_chw: torch.Tensor, mean, std) -> np.ndarray:
    """[3,H,W] 归一化张量 -> [H,W,3] uint8。"""
    x = img_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    x = x * np.asarray(std, np.float32) + np.asarray(mean, np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _mask_vis(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask).astype(np.uint8) * 255)[..., None].repeat(3, axis=2)


def _error_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP 绿色 / FP 红色 / FN 蓝色 的误差图。"""
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    canvas = np.zeros((*gt.shape, 3), np.uint8)
    canvas[gt & pred] = (0, 200, 0)      # TP
    canvas[~gt & pred] = (255, 0, 0)     # FP
    canvas[gt & ~pred] = (0, 0, 255)     # FN
    return canvas


def _panel(img_hw3: np.ndarray, title: str, title_h: int = 18) -> np.ndarray:
    strip = np.full((title_h, img_hw3.shape[1], 3), 255, np.uint8)
    canvas = np.concatenate([strip, img_hw3], axis=0)
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((4, 3), title, fill=(0, 0, 0))
    return np.asarray(im)


def save_visualization(a, b, gt, pred, path: str, mean=None, std=None):
    """保存单样本四联图（含误差图），共 5 栏。

    a, b: [3,H,W] 归一化张量；gt, pred: [H,W] 0/1 数组。
    """
    if mean is None or std is None:
        mean, std = resolve_norm("imagenet")
    panels = [
        _panel(_denorm(a, mean, std), "A"),
        _panel(_denorm(b, mean, std), "B"),
        _panel(_mask_vis(gt), "GT"),
        _panel(_mask_vis(pred), "Pred"),
        _panel(_error_overlay(gt, pred), "TP/FP/FN"),
    ]
    row = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(row).save(path)


def visualize_batch(a, b, gt, pred, out_dir: str, names, start_idx: int, max_items: int,
                    mean=None, std=None):
    """对 batch 中前若干样本保存可视化，返回下一个索引。"""
    saved = 0
    bs = a.shape[0]
    for i in range(bs):
        if saved >= max_items:
            break
        name = names[i] if isinstance(names, (list, tuple)) else str(names[i])
        save_visualization(
            a[i], b[i], gt[i], pred[i],
            os.path.join(out_dir, f"{start_idx + saved:04d}_{name}.png"),
            mean, std,
        )
        saved += 1
    return start_idx + saved
