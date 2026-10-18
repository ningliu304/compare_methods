"""ChangeViT 适配器。

网络: model/trainer.py::Trainer，forward(x, y) 取 A/B 两个张量，返回 sigmoid 概率 [B,1,H,W]。
损失: model/utils.py::BCEDiceLoss（BCE + 1 - Dice，输入为 sigmoid 概率）。
二值化: prob > 0.5

注意:
- 原 Encoder 硬编码 ViT token 的 h=w=16（对应 256 输入）；本适配器在 build_model 后
  重绑 encoder.forward，按实际 token 数动态计算 h/w，使其支持 512 输入（ViT 的
  pos_embed 已内置插值，无需改动）。
- 原仓库用 cv2 读图（BGR）且归一化用 BGR 顺序的 ImageNet 均值，故这里把 RGB 翻成 BGR，
  与原文保持一致。
- 需要把预训练权重放到 ChangeViT-master/checkpoint/ 下（small: dinov2_vits14_pretrain.pth；
  tiny: deit_tiny_patch16_224-a1311bcf.pth）。
"""

import os
import shutil
import types

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to

HERE = os.path.dirname(os.path.abspath(__file__))

# 各 model_type 需要的预训练 ViT 权重（仓库内相对路径）
VIT_CKPT = {
    "tiny": "checkpoint/deit_tiny_patch16_224-a1311bcf.pth",
    "small": "checkpoint/dinov2_vits14_pretrain.pth",
}
VIT_CKPT_URL = {
    "tiny": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    "small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
}


def _encoder_forward_dynamic(self, x, y):
    """按输入尺寸动态计算 token 网格的 h/w，替换原硬编码 h=w=16。"""
    from einops import rearrange

    v_x = self.vit(x)
    v_y = self.vit(y)
    h = w = int(round(v_x.shape[1] ** 0.5))
    v_x = rearrange(v_x, "b (h w) c -> b c h w", h=h, w=w)
    v_y = rearrange(v_y, "b (h w) c -> b c h w", h=h, w=w)
    c_x = self.detail_capture(x)
    c_y = self.detail_capture(y)
    return c_x + [v_x], c_y + [v_y]


class ChangeViTAdapter(CDAdapter):
    name = "ChangeViT"

    def __init__(self, model_type: str = "small"):
        self.model_type = model_type

    def _ensure_ckpt(self):
        """确保预训练 ViT 权重就位；缺失时给出明确指引或按环境变量复制。"""
        rel = VIT_CKPT[self.model_type]
        dst = os.path.join(HERE, rel)
        if os.path.isfile(dst):
            return
        src = os.environ.get("CHANGEVIT_VIT_CKPT")
        if src and os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"[ChangeViT] 已将 {src} 复制为 {dst}")
            return
        raise FileNotFoundError(
            f"ChangeViT 缺少预训练 ViT 权重: {dst}\n"
            f"请下载后放到该路径，或设环境变量 CHANGEVIT_VIT_CKPT=<权重路径>。\n"
            f"下载地址: {VIT_CKPT_URL[self.model_type]}"
        )

    def build_model(self, cfg):
        self._ensure_ckpt()
        cwd = os.getcwd()
        os.chdir(HERE)   # 仓库内权重路径为相对路径 "checkpoint/..."
        try:
            from model.trainer import Trainer

            model = Trainer(self.model_type)
        finally:
            os.chdir(cwd)
        # 支持非 256 输入
        model.encoder.forward = types.MethodType(_encoder_forward_dynamic, model.encoder)
        return model

    def prepare_batch(self, batch, device):
        # RGB -> BGR（匹配原仓库 cv2 读取）
        a = batch["a"].to(device, non_blocking=True).float().flip(1)
        b = batch["b"].to(device, non_blocking=True).float().flip(1)
        target = batch["label"].to(device, non_blocking=True).long()
        return {"a": a, "b": b}, target

    def forward(self, model, inputs):
        prob = model(inputs["a"], inputs["b"])   # [B,1,H,W] sigmoid
        return {"prob": prob}

    def compute_loss(self, model, out, target):
        from model.utils import BCEDiceLoss

        prob = resize_to(out["prob"], target)
        tgt = target.float().unsqueeze(1)        # [B,1,H,W]
        return BCEDiceLoss(prob.clamp(1e-6, 1 - 1e-6), tgt)

    def predict_binary(self, out, inputs):
        return (out["prob"].squeeze(1) > 0.5).long()
