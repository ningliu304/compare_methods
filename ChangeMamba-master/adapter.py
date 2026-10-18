"""ChangeMamba 适配器。

网络: changedetection/models/ChangeMambaBCD.py::ChangeMambaBCD
      forward(pre, post) 取 A/B 两个张量，返回 2 通道 logits → resize 回输入尺寸。
损失: 与原实现一致 = CrossEntropy + 0.75 * Lovasz-Softmax。
二值化: argmax

说明:
- 原框架是 VMamba 派生的独立工程（自有迭代式训练循环、T1/T2/GT 数据集），这里只用它的
  模型定义，训练走统一的 common 引擎，数据用统一数据集（RGB + ImageNet 归一化，
  与其 imutils.normalize_img 的 0~255 尺度 ImageNet 均值等价）。
- 需要先编译 SSM 自定义算子：`cd kernels/selective_scan && pip install .`
  （否则 import 时会报 selective_scan_cuda 缺失）。
- 可选加载 VMamba 预训练编码器：设环境变量 CHANGEMAMBA_PRETRAINED=/path/to/vssm_ckpt.pth。
"""

import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

from common.adapter import CDAdapter, resize_to

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_CFG = os.path.join(
    HERE, "changedetection", "configs", "vssm1", "vssm_tiny_224_0229flex.yaml")


class ChangeMambaAdapter(CDAdapter):
    name = "ChangeMamba"

    def __init__(self, config_file: str = None, pretrained: str = None):
        self.config_file = config_file or DEFAULT_CFG
        self.pretrained = pretrained if pretrained is not None else os.environ.get("CHANGEMAMBA_PRETRAINED")

    def build_model(self, cfg):
        try:
            from changedetection.configs.config import get_config
            from changedetection.script.script_utils import get_vssm_kwargs
            from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "无法导入 ChangeMamba（需要已编译的 SSM 算子）。\n"
                "请先执行: cd kernels/selective_scan && pip install .\n"
                f"原始错误: {e}"
            )

        args = Namespace(cfg=self.config_file, opts=None)
        config = get_config(args)
        return ChangeMambaBCD(pretrained=self.pretrained, **get_vssm_kwargs(config))

    def forward(self, model, inputs):
        logits = model(inputs["a"], inputs["b"])   # [B,2,H,W]
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        return {"logits": logits, "prob": prob}

    def compute_loss(self, model, out, target):
        from changedetection.utils_func import lovasz_loss as L

        logits = resize_to(out["logits"], target)
        tgt = target.long()
        ce = F.cross_entropy(logits, tgt, ignore_index=255)
        lov = L.lovasz_softmax(F.softmax(logits, dim=1), tgt, ignore=255)
        return ce + 0.75 * lov

    def predict_binary(self, out, inputs):
        return out["logits"].argmax(dim=1).long()
