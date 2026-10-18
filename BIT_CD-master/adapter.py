"""BIT_CD 适配器：复用仓库原始网络定义，接入统一训练引擎。

网络: models/networks.py::define_G  输出 [B,2,H,W] logits
损失: 交叉熵；二值化: argmax
"""

from argparse import Namespace

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to


class BITAdapter(CDAdapter):
    name = "BIT_CD"
    default_model = "base_transformer_pos_s4_dd8"

    def build_model(self, cfg):
        from models.networks import define_G

        args = Namespace(
            net_G=cfg.model or self.default_model,
            gpu_ids=[],           # 由 engine 统一搬到 device
            n_class=cfg.n_class,
        )
        return define_G(args=args, gpu_ids=[])

    def forward(self, model, inputs):
        logits = model(inputs["a"], inputs["b"])            # [B,2,H,W]
        prob = torch.softmax(logits, dim=1)[:, 1:2]         # 变化类概率
        return {"logits": logits, "prob": prob}

    def compute_loss(self, model, out, target):
        logits = resize_to(out["logits"], target)
        return F.cross_entropy(logits, target.long())

    def predict_binary(self, out, inputs):
        return out["logits"].argmax(dim=1).long()
