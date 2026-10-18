"""TinyCD 适配器。

网络: models/change_classifier.py::ChangeClassifier，forward(ref, test) 输出 (B,1,H,W)
      且内部已含 Sigmoid()，即直接为变化概率。
损失: BCE（与原实现一致）；二值化: prob > 0.5
"""

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to


class TinyCDAdapter(CDAdapter):
    name = "TinyCD"

    def build_model(self, cfg):
        from models.change_classifier import ChangeClassifier

        return ChangeClassifier(pretrained=True)

    def prepare_batch(self, batch, device):
        inputs = {
            "a": batch["a"].to(device, non_blocking=True).float(),
            "b": batch["b"].to(device, non_blocking=True).float(),
        }
        target = batch["label"].to(device, non_blocking=True).float()   # BCE 需要 float
        return inputs, target

    def forward(self, model, inputs):
        prob = model(inputs["a"], inputs["b"])   # [B,1,H,W] sigmoid 概率
        return {"prob": prob}

    def compute_loss(self, model, out, target):
        prob = resize_to(out["prob"], target)
        tgt = target.unsqueeze(1).float()        # [B,1,H,W]
        return F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), tgt)

    def predict_binary(self, out, inputs):
        return (out["prob"].squeeze(1) > 0.5).long()
