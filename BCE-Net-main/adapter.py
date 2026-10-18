"""BCE-Net 适配器（双时相二值 CD 改造版）。

网络: model_bcd.py::Baseline34BCD(A, B) -> [B,1,H,W] logits
损失: BCE + Dice（EVLab 变化像素仅约 2.8%，纯 BCE 会塌缩到全背景）
二值化: sigmoid(logit) > 0.5
"""

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to


class BCENetAdapter(CDAdapter):
    name = "BCE-Net"

    def build_model(self, cfg):
        from model_bcd import Baseline34BCD

        return Baseline34BCD(num_classes=1, pretrained=True)

    def forward(self, model, inputs):
        logits = model(inputs["a"], inputs["b"])       # [B,1,H,W]
        return {"logits": logits, "prob": torch.sigmoid(logits)}

    def compute_loss(self, model, out, target):
        logits = resize_to(out["logits"], target).squeeze(1)   # [B,H,W]
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1 - (2 * inter + 1.0) / (union + 1.0)
        return bce + dice.mean()

    def predict_binary(self, out, inputs):
        return (torch.sigmoid(out["logits"]).squeeze(1) > 0.5).long()
