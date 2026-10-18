"""DSAMNet 适配器。

网络: model/dsamnet.py::DSAMNet(2)，forward(input1, input2) 返回 (dist, ds2, ds3)。
      dist 为成对距离图（非概率），原实现在 dist > 1 处判为变化。
损失: 与原实现一致 = BCL(dist, label) + wDice * 0.5*(Dice(ds2)+Dice(ds3))。
二值化: 保留原设计 dist > threshold（默认 1.0，可通过 --extra 或直接改这里调整）。
"""

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to

from loss.BCL import BCL
from loss.DiceLoss import DiceLoss


class DSAMNetAdapter(CDAdapter):
    name = "DSAMNet"

    def __init__(self, threshold: float = 1.0, w_dice: float = 0.1):
        self.threshold = threshold
        self.w_dice = w_dice
        self.bcl = BCL()
        self.dice = DiceLoss()

    def build_model(self, cfg):
        from model.dsamnet import DSAMNet

        return DSAMNet(cfg.n_class)

    def forward(self, model, inputs):
        dist, ds2, ds3 = model(inputs["a"], inputs["b"])
        prob = torch.sigmoid(dist)                    # 仅供可视化，概率化
        return {"dist": dist, "ds2": ds2, "ds3": ds3, "prob": prob}

    def compute_loss(self, model, out, target):
        tgt1 = target.unsqueeze(1).float()                          # [B,1,H,W]
        onehot = F.one_hot(target.long(), 2).permute(0, 3, 1, 2).float()  # [B,2,H,W]

        ct = self.bcl(out["dist"], tgt1.clone())                    # BCL 会原地改 label
        d2 = self.dice(resize_to(out["ds2"], onehot), onehot)
        d3 = self.dice(resize_to(out["ds3"], onehot), onehot)
        dice_loss = 0.5 * (d2 + d3)
        return ct + self.w_dice * dice_loss

    def predict_binary(self, out, inputs):
        dist = out["dist"]
        return (dist > self.threshold).squeeze(1).long()
