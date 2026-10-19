"""SAM-CD 适配器。

网络: models/SAM_CD.py::SAM_CD（FastSAM-x 冻结骨干 + 适配器/解码器）
输出: (change_logits, aux_a, aux_b)，均为 [B,C,H,W]
损失: BCEWithLogits(change) + LatentSimilarity(aux_a, aux_b)
二值化: sigmoid(change_logits) > 0.5
"""

import os

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to


class SAMCDAdapter(CDAdapter):
    name = "SAM-CD"

    def __init__(self):
        self._sem_loss = None

    def build_model(self, cfg):
        from models.SAM_CD import SAM_CD

        weight = os.environ.get(
            "SAMCD_WEIGHTS",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "FastSAM-x.pt"))
        # 本地没有权重时交给 ultralytics 按名称自动下载
        if not os.path.isfile(weight):
            weight = "FastSAM-x.pt"
        return SAM_CD(device=cfg.device, model_name=weight)

    def forward(self, model, inputs):
        logits, aux_a, aux_b = model(inputs["a"], inputs["b"])
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "aux_a": aux_a,
            "aux_b": aux_b,
        }

    def compute_loss(self, model, out, target):
        from utils.loss import LatentSimilarity

        logits = resize_to(out["logits"], target).squeeze(1)
        target_f = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target_f)
        if self._sem_loss is None:
            self._sem_loss = LatentSimilarity(T=3.0).to(logits.device)
        sem = self._sem_loss(out["aux_a"], out["aux_b"],
                             target_f.unsqueeze(1))
        return bce + sem

    def predict_binary(self, out, inputs):
        return (torch.sigmoid(out["logits"]).squeeze(1) > 0.5).long()
