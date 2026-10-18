"""checkpoint 工具：支持按目录枚举候选权重 + 跨格式加载。

用于 MMSeg 系模型（CMCDNet/ACTNet）：训练按间隔保存多个 epoch 权重，
再由统一评估脚本按「验证集变化类 F1」挑选最优，保证与统一引擎口径一致。
"""

import glob
import os

import torch


def resolve_checkpoints(path: str):
    """path 为单个 .pth 时返回 [path]；为目录时返回其下所有 .pth（按名排序）。"""
    if os.path.isdir(path):
        ckpts = sorted(glob.glob(os.path.join(path, "*.pth")))
        if not ckpts:
            raise FileNotFoundError(f"目录 {path} 下未找到 .pth 权重")
        return ckpts
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到权重: {path}")
    return [path]


def load_weights(model, ckpt_path: str, device="cpu") -> None:
    """加载权重，兼容 mmseg 0.x 的 {'state_dict': ...} 与 mmseg 1.x 的裸 state_dict。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
