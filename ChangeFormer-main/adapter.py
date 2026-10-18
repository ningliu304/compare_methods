"""ChangeFormer 适配器：复用仓库 models/networks.py::define_G。

网络: ChangeFormerV6(embed_dim=64)，forward 返回多尺度列表，取 [-1] 为全分辨率 logits [B,2,H,W]
损失: 交叉熵；二值化: argmax
"""

from argparse import Namespace

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to


class ChangeFormerAdapter(CDAdapter):
    name = "ChangeFormer"
    default_model = "ChangeFormerV6"

    def __init__(self, embed_dim: int = 256):
        # 与原论文一致：embed_dim=256 → 约 41M 参数
        self.embed_dim = embed_dim

    def build_model(self, cfg):
        from models.networks import define_G

        args = Namespace(
            net_G=cfg.model or self.default_model,
            embed_dim=self.embed_dim,
            gpu_ids=[],
            n_class=cfg.n_class,
        )
        return define_G(args=args, gpu_ids=[])

    def forward(self, model, inputs):
        preds = model(inputs["a"], inputs["b"])   # list of multi-scale logits
        logits = preds[-1] if isinstance(preds, (list, tuple)) else preds
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        return {"logits": logits, "prob": prob}

    def compute_loss(self, model, out, target):
        logits = resize_to(out["logits"], target)
        return F.cross_entropy(logits, target.long())

    def predict_binary(self, out, inputs):
        return out["logits"].argmax(dim=1).long()
