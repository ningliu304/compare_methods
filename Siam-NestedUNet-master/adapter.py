"""Siam-NestedUNet (SNUNet-CD) 适配器。

网络: models/Models.py::SNUNet_ECAM(3, 2)，forward(xA, xB) 返回 (out,)，out 为
      [B,2,H,W] logits。
损失: 与原实现一致的 hybrid loss = FocalLoss(gamma=0, alpha=None) + dice_loss，
      逐输出求和。这里从 utils.metrics 直接取组件重建（utils.losses 在 import 时会
      解析 argv，与统一 CLI 冲突，故不直接使用）。
二值化: argmax
"""

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to

from utils.metrics import FocalLoss


def _dice_loss(logits, true, eps=1e-7):
    """与 utils.metrics.dice_loss 等价，但 one-hot 的 eye 建在 logits 所在设备上。

    原实现用 torch.eye(num_classes)（CPU）去索引 GPU 上的 target，改为按设备创建。
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=logits.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        probas = torch.cat([pos_prob, 1 - pos_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=logits.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    return 1 - (2. * intersection / (cardinality + eps)).mean()


def _hybrid_loss(predictions, target):
    """与 utils/losses.py::hybrid_loss 等价（FocalLoss(gamma=0) + Dice）。"""
    loss = 0
    focal = FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:
        loss = loss + focal(prediction, target) + _dice_loss(prediction, target)
    return loss


class SNUNetAdapter(CDAdapter):
    name = "Siam-NestedUNet"

    def build_model(self, cfg):
        from models.Models import SNUNet_ECAM

        return SNUNet_ECAM(3, 2)   # (in_ch=3, n_class=2)

    def forward(self, model, inputs):
        preds = model(inputs["a"], inputs["b"])     # (out,) 1-tuple
        logits = preds[-1]
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        return {"logits": logits, "preds": preds, "prob": prob}

    def compute_loss(self, model, out, target):
        preds = out["preds"]
        # 深监督需要各输出与 target 对齐
        aligned = [resize_to(p, target) for p in preds]
        return _hybrid_loss(aligned, target.long())

    def predict_binary(self, out, inputs):
        return out["logits"].argmax(dim=1).long()
