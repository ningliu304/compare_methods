"""模型适配器基类。

每个对比模型只需实现 5 个钩子，训练/验证/测试/指标/可视化/checkpoint 由
common.engine.Trainer 统一完成。

重要约定：二值化规则（argmax 还是 sigmoid>0.5 还是距离阈值）只在 adapter 内实现，
不同模型的决策规则由其原论文设计决定，需在论文中注明。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_to(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """当 logits 空间尺寸与 target 不一致时双线性上采样对齐。"""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


class CDAdapter:
    name = "base"

    def build_model(self, cfg) -> nn.Module:
        """构建并返回网络（不要 to(device)，engine 会处理）。"""
        raise NotImplementedError

    def prepare_batch(self, batch, device):
        """把 dataloader 的 batch 转为 (inputs, target)。engine 只负责搬运到 device。

        默认返回 inputs={'a','b'}（float），target=long [B,H,W]。
        使用 sigmoid 损失的模型可在子类中把 target 转为 float。
        """
        inputs = {
            "a": batch["a"].to(device, non_blocking=True).float(),
            "b": batch["b"].to(device, non_blocking=True).float(),
        }
        target = batch["label"].to(device, non_blocking=True).long()
        return inputs, target

    def forward(self, model, inputs) -> dict:
        """前向。需返回至少含 'prob' 的字典：'prob' 为 [B,1,H,W] 变化概率(0~1)。

        可附带 'logits' / 'aux' 供 loss 与二值化使用。
        """
        raise NotImplementedError

    def compute_loss(self, model, out, target) -> torch.Tensor:
        raise NotImplementedError

    def predict_binary(self, out, inputs) -> torch.Tensor:
        """返回 [B,H,W] long 的二值图（0/1）。"""
        raise NotImplementedError
