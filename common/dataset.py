"""统一变化检测数据集。

数据目录结构::

    <root>/
      train/{A, B, label}
      val/{A, B, label}
      test/{A, B, label}

- A、B 为双时相 RGB 图像，label 为单通道二值掩膜（0 背景 / 非 0 变化）。
- 同一 split 内按文件 stem 配对（A/B/label 文件名主干一致，扩展名可不同）。
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.config import resolve_norm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
# 兼容不同仓库的 label 目录命名
LABEL_DIR_CANDIDATES = ("label", "labels", "Label", "L", "OUT", "gt", "GT", "mask", "masks")


def _list_stems(folder: str):
    stems = {}
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXTS:
            stems[stem] = fn  # 主干 -> 实际文件名
    return stems


def _resolve_label_dir(split_dir: str) -> str:
    for cand in LABEL_DIR_CANDIDATES:
        p = os.path.join(split_dir, cand)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"在 {split_dir} 下找不到 label 目录，尝试过: {LABEL_DIR_CANDIDATES}"
    )


def _explain_missing_a(split_dir: str, a_dir: str) -> str:
    """A 目录缺失时给出可操作的提示（常见于把「数据集根目录」当成了单个数据集）。"""
    root = os.path.dirname(os.path.normpath(split_dir))
    try:
        from common.datasets import discover_datasets

        found = discover_datasets(root)
        if found:
            names = ", ".join(found)
            return (
                f"找不到 A 目录: {a_dir}\n"
                f"看起来 {root} 是「数据集根目录」，其下包含多个数据集。\n"
                f"请用 --dataset <名称> 选择，例如: --dataset {next(iter(found))}\n"
                f"可选数据集: {names}"
            )
    except Exception:  # noqa: BLE001
        pass
    return (
        f"找不到 A 目录: {a_dir}\n"
        f"请确认 <数据集>/<split>/{{A,B,label}} 存在，例如 data/Evlab/train/A。"
    )


class CDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512,
                 augment: bool = False, norm: str = "imagenet"):
        self.split_dir = os.path.join(root, split)
        self.a_dir = os.path.join(self.split_dir, "A")
        self.b_dir = os.path.join(self.split_dir, "B")
        if not os.path.isdir(self.a_dir):
            raise FileNotFoundError(_explain_missing_a(self.split_dir, self.a_dir))
        self.l_dir = _resolve_label_dir(self.split_dir)

        self.a_stems = _list_stems(self.a_dir)
        self.b_stems = _list_stems(self.b_dir)
        self.l_stems = _list_stems(self.l_dir)

        # 取 A/B/label 三者共有的主干名，保持顺序稳定
        self.names = [n for n in sorted(self.a_stems) if n in self.b_stems and n in self.l_stems]
        if len(self.names) == 0:
            raise RuntimeError(f"{split} 集未找到 A/B/label 三者配对的样本于 {self.split_dir}")

        self.img_size = img_size
        self.augment = augment
        self.mean, self.std = resolve_norm(norm)

    def __len__(self):
        return len(self.names)

    def _load_rgb(self, folder, stems, name):
        img = Image.open(os.path.join(folder, stems[name])).convert("RGB")
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _load_label(self, name):
        img = Image.open(os.path.join(self.l_dir, self.l_stems[name])).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.NEAREST)
        arr = np.asarray(img)
        # 兼容 0/1 与 0/255 两种二值标注
        if arr.size and arr.max() <= 1:
            label = arr.astype(np.int64)
        else:
            label = (arr > 127).astype(np.int64)
        return label

    def _to_tensor(self, arr_hwc_float):
        arr = arr_hwc_float.astype(np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        name = self.names[index]
        a = np.asarray(self._load_rgb(self.a_dir, self.a_stems, name), dtype=np.float32)
        b = np.asarray(self._load_rgb(self.b_dir, self.b_stems, name), dtype=np.float32)
        label = self._load_label(name)

        if self.augment:
            # 统一增强：随机水平/垂直翻转 + 90° 旋转
            if random.random() < 0.5:
                a, b, label = a[:, ::-1], b[:, ::-1], label[:, ::-1]
            if random.random() < 0.5:
                a, b, label = a[::-1, :], b[::-1, :], label[::-1, :]
            k = random.randint(0, 3)
            if k:
                a = np.rot90(a, k)
                b = np.rot90(b, k)
                label = np.rot90(label, k)
            a = np.ascontiguousarray(a)
            b = np.ascontiguousarray(b)
            label = np.ascontiguousarray(label)

        return {
            "a": self._to_tensor(a),
            "b": self._to_tensor(b),
            "label": torch.from_numpy(label),
            "name": name,
        }


def build_loaders(cfg) -> dict:
    """按 cfg 构建 train/val/test DataLoader。"""
    augment = True
    train_set = CDDataset(cfg.root, "train", cfg.img_size, augment, cfg.norm)
    val_set = CDDataset(cfg.root, "val", cfg.img_size, False, cfg.norm)
    test_set = CDDataset(cfg.root, "test", cfg.img_size, False, cfg.norm)

    common = dict(num_workers=cfg.num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **common),
        "val": DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common),
        "test": DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, **common),
    }
