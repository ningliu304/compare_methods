# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from opencd.registry import MODELS


@MODELS.register_module()
class FeatureFusionAdapterNeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates 
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                 cnn_channels=None,
                 tf_channels=None,
                 out_channels=None,
                 adapter=True,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.tf_channels = tf_channels
        self.channels = out_channels
        self.out_indices = out_indices
        self.adapter = adapter

        self.cnn_adapter = nn.ModuleList()
        self.tf_adapter = nn.ModuleList()
        for i in range(len(cnn_channels)):
            self.cnn_adapter.append(Adapter(d_model=cnn_channels[i], out_dim=out_channels[i],
                                            bottleneck=cnn_channels[i] // 8,
                                            adapter_layernorm_option='out'))
        for i in range(len(tf_channels)):
            self.tf_adapter.append(Adapter(d_model=tf_channels[i], out_dim=out_channels[i],
                                           bottleneck=tf_channels[i] // 8,
                                           adapter_layernorm_option='out'))

        self.cnn_rec = nn.ModuleList()
        self.tf_rec = nn.ModuleList()
        for i in range(len(cnn_channels)):
            self.cnn_rec.append(nn.Conv2d(in_channels=cnn_channels[i], out_channels=out_channels[i],
                                              kernel_size=1))
        for i in range(len(tf_channels)):
            self.tf_rec.append(nn.Conv2d(in_channels=tf_channels[i], out_channels=out_channels[i],
                                             kernel_size=1))

    def spatial_alignment(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):

        if tf_feat_from.shape[-2:] != cnn_feat_from.shape[-2:]:
            tf_feat_from = F.interpolate(tf_feat_from, size=cnn_feat_from.shape[-2:])
            tf_feat_to = F.interpolate(tf_feat_to, size=cnn_feat_from.shape[-2:])

        return cnn_feat_from, cnn_feat_to, tf_feat_from, tf_feat_to

    @staticmethod
    def fusion(x1, x2, x3, x4):
        """Specify the form of feature fusion"""

        x_from = torch.cat([x1, x2], dim=1)
        x_to = torch.cat([x3, x4], dim=1)

        return x_from, x_to

    def forward_adapter(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):
        outs_from = []
        outs_to = []
        for i in range(len(cnn_feat_from)):
            x1 = self.cnn_adapter[i](cnn_feat_from[i])
            x2 = self.tf_adapter[i](tf_feat_from[i])
            x3 = self.cnn_adapter[i](cnn_feat_to[i])
            x4 = self.tf_adapter[i](tf_feat_to[i])

            x1, x2, x3, x4 = self.spatial_alignment(x1, x2, x3, x4)
            out_from, out_to = self.fusion(x1, x2, x3, x4)
            outs_from.append(out_from)
            outs_to.append(out_to)

        outs_from = [outs_from[i] for i in self.out_indices]
        outs_to = [outs_to[i] for i in self.out_indices]
        return tuple(outs_from), tuple(outs_to)

    def forward_no_adapter(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):
        outs_from = []
        outs_to = []
        for i in range(len(cnn_feat_from)):
            x1 = self.cnn_rec[i](cnn_feat_from[i])
            x2 = self.tf_rec[i](tf_feat_from[i])
            x3 = self.cnn_rec[i](cnn_feat_to[i])
            x4 = self.tf_rec[i](tf_feat_to[i])

            x1, x2, x3, x4 = self.spatial_alignment(x1, x2, x3, x4)
            out_from, out_to = self.fusion(x1, x2, x3, x4)
            outs_from.append(out_from)
            outs_to.append(out_to)

        outs_from = [outs_from[i] for i in self.out_indices]
        outs_to = [outs_to[i] for i in self.out_indices]
        return tuple(outs_from), tuple(outs_to)

    def forward(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):
        """Forward function."""
        if self.adapter:
            outs_from, outs_to = self.forward_adapter(cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to)
        else:
            outs_from, outs_to = self.forward_no_adapter(cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to)
        return outs_from, outs_to


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_dim=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 use_square=False, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.use_square = use_square
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(out_dim)
        self.scale = nn.Linear(self.n_embd, 1)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        if out_dim is None:
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        else:
            self.up_proj = nn.Linear(self.down_size, out_dim)

        self.dropout = dropout

    def forward(self, x, add_residual=False, residual=None):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        scale = F.relu(self.scale(x))
        down = self.down_proj(x)
        # down = down.permute(0, 2, 3, 1).reshape(b, (h*w), -1)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        # down = down.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        up = self.up_proj(down)
        up = up * scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            up = up.permute(0, 3, 1, 2)
            output = up + residual
        else:
            up = up.permute(0, 3, 1, 2)
            output = up
        return output
