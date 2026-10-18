"""BCE-Net (Baseline34) 的双时相二值变化检测改造版。

原始 BCE-Net 为单时相“建筑语义变化”模型：输入单张影像 + 旧建筑先验掩膜，
输出 {旧建筑/新建/迁移} 三个分支。本改造用于统一对比实验（双时相 RGB -> 二值变化）：

- 保留其 resnet34 编码器几何 与 U 形编解码跳连结构（与原始 Baseline34 一致）；
- 移除“旧建筑先验 / 新-迁移分支”，改为对 A、B 两时相特征做差分融合；
- 输出单通道变化 logits [B,1,H,W]。

说明:
- DCNv2 为可选项：若已编译则使用原可变形卷积 DCN，否则自动退化为普通 3x3 卷积，
  保证在未编译 DCNv2 的环境也能运行（原仓库 CDResSIBU.py 在 import 时强制依赖 DCNv2）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:  # DCNv2 为可选项
    from DCNv2.dcn_v2 import DCN
    _HAS_DCN = True
except Exception:  # noqa: BLE001
    DCN = None
    _HAS_DCN = False


class SEModule(nn.Module):
    """Squeeze-and-Excitation（与原 Baseline34 一致）。"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m = self.avg_pool(x)
        m = self.sigmoid(self.fc2(self.relu(self.fc1(m))))
        return x * m


class _DeformConv(nn.Module):
    """有 DCNv2 用可变形卷积，否则退化普通卷积。"""

    def __init__(self, ch):
        super().__init__()
        if _HAS_DCN:
            self.op = DCN(ch, ch, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)
        else:
            self.op = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.op(x)


class Baseline34BCD(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 use_dcn=True, use_se=True):
        super().__init__()
        # 与 BCE-Net 相同的 resnet34 编码器
        self.backbone = torchvision.models.resnet34(pretrained=pretrained)

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        # 双时相差分后的 1x1 融合
        self.fuse1 = nn.Conv2d(64, 64, 1)
        self.fuse2 = nn.Conv2d(128, 128, 1)
        self.fuse3 = nn.Conv2d(256, 256, 1)
        self.fuse4 = nn.Conv2d(512, 512, 1)

        # U 形解码器（通道数与原始 Baseline34 对齐）
        self.deconv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.Conv2d(384, 192, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(192)
        self.deconv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.sel = SEModule(128, reduction=2) if use_se else nn.Identity()
        self.dcn = _DeformConv(128) if use_dcn else nn.Identity()

        self.finalseg = nn.Conv2d(128, num_classes, 3, padding=1)

    def _encode(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        e1 = self.backbone.layer1(x)      # 1/4,  64
        e2 = self.backbone.layer2(e1)     # 1/8,  128
        e3 = self.backbone.layer3(e2)     # 1/16, 256
        e4 = self.backbone.layer4(e3)     # 1/32, 512
        e1 = self.bn1(self.conv1(e1))
        e2 = self.bn2(self.conv2(e2))
        e3 = self.bn3(self.conv3(e3))
        e4 = self.bn4(self.conv4(e4))
        return e1, e2, e3, e4

    def forward(self, a, b):
        a1, a2, a3, a4 = self._encode(a)
        b1, b2, b3, b4 = self._encode(b)

        # 双时相特征差分融合
        f1 = self.fuse1(torch.abs(a1 - b1))
        f2 = self.fuse2(torch.abs(a2 - b2))
        f3 = self.fuse3(torch.abs(a3 - b3))
        f4 = self.fuse4(torch.abs(a4 - b4))

        d1 = self.norm1(self.deconv1(f4))
        d1 = F.interpolate(d1, f3.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.norm2(self.deconv2(torch.cat([d1, f3], dim=1)))
        d2 = F.interpolate(d2, f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.norm3(self.deconv3(torch.cat([d2, f2], dim=1)))
        d3 = F.interpolate(d3, f1.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.norm4(self.deconv4(torch.cat([d3, f1], dim=1)))
        d4 = self.sel(self.dcn(d4))
        d4 = F.interpolate(d4, a.shape[2:], mode="bilinear", align_corners=False)
        return self.finalseg(d4)          # [B,1,H,W] logits
