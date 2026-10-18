import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmengine.model import BaseModule
from mmcv.ops.carafe import normal_init, xavier_init, carafe
from mmcv.cnn import ConvModule
from opencd.registry import MODELS
import torch.fft

from ..utils.agent_attention import AgentAttention, CrossAgentAttention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class FFTAttn(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm, h=14, w=8, drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.norm2 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        b, c, h, w = x.shape
        # B N C
        x = x.reshape(b, c, -1).permute(0, 2, 1)

        x = self.filter(self.norm1(x))
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.norm2(x)

        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = self.norm(x_out)
        return x_out


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, cnn_channel=64, tf_channel=64, channels=64, h=48, w=48, sr_ratio=4):
        super(AFF, self).__init__()

        self.h = h
        self.w = w

        self.detail_att_freq = FFTAttn(dim=channels, h=h, w=(w // 2) + 1)
        self.detail_att_spatial = FocalModulation(dim=channels)
        self.drop_detail = nn.Dropout(0.1)
        self.mlp_detail = Mlp(in_features=2 * channels, hidden_features=channels * 4, out_features=channels)
        self.norm_detail_freq = nn.LayerNorm(channels)
        self.norm_detail_spatial = nn.LayerNorm(channels)
        self.proj_spatial_detail = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_freq_detail = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm_detail_freq_post = nn.LayerNorm(channels)
        self.norm_detail_spatial_post = nn.LayerNorm(channels)

        self.sem_att_freq = FFTAttn(dim=channels, h=h, w=(w // 2) + 1)
        self.sem_att_spatial = AgentAttention(dim=channels, num_patches=h * w)
        self.mlp_sem = Mlp(in_features=2 * channels, hidden_features=channels * 4, out_features=channels)
        self.norm_sem_freq = nn.LayerNorm(channels)
        self.norm_sem_spatial = nn.LayerNorm(channels)
        self.drop_sem = nn.Dropout(0.1)
        self.proj_spatial_sem = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_freq_sem = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm_sem_freq_post = nn.LayerNorm(channels)
        self.norm_sem_spatial_post = nn.LayerNorm(channels)

        self.cross_fusion_detail_freq = CrossAgentAttention(dim=channels, num_patches=h * w, sr_ratio=sr_ratio)
        self.cross_fusion_detail_spatial = CrossAgentAttention(dim=channels, num_patches=h * w, sr_ratio=sr_ratio)
        self.cross_fusion_sem_freq = CrossAgentAttention(dim=channels, num_patches=h * w, sr_ratio=sr_ratio)
        self.cross_fusion_sem_spatial = CrossAgentAttention(dim=channels, num_patches=h * w, sr_ratio=sr_ratio)

        self.proj_cnn = nn.Conv2d(cnn_channel, channels, 1, 1)
        self.proj_tf = nn.Conv2d(tf_channel, channels, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def local_attn_spatial_only(self, x):
        b, c, h, w = x.shape

        x_spatial = self.detail_att_spatial(x.permute(0, 2, 3, 1))
        x_spatial = x_spatial.permute(0, 3, 1, 2)
        x_spatial = self.proj_spatial_detail(x_spatial)
        x_spatial = x_spatial.reshape(b, c, -1).permute(0, 2, 1)

        out = self.drop_detail(self.drop_detail(self.mlp_detail(x_spatial)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def local_attn_freq_only(self, x):
        b, c, h, w = x.shape
        x_freq = self.detail_att_freq(x)
        x_freq = self.proj_freq_detail(x_freq)
        x_freq = x_freq.reshape(b, c, -1).permute(0, 2, 1)

        out = self.drop_detail(self.drop_detail(self.mlp_detail(x_freq)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def local_att(self, x):
        b, c, h, w = x.shape

        x_freq = self.detail_att_freq(x)
        x_freq = self.proj_freq_detail(x_freq)
        x_freq = x_freq.reshape(b, c, -1).permute(0, 2, 1)
        x_freq = self.norm_detail_freq(x_freq)

        x_spatial = x.permute(0, 2, 3, 1)
        x_spatial = self.norm_detail_spatial(x_spatial)
        x_spatial = self.detail_att_spatial(x_spatial)
        x_spatial = x_spatial.permute(0, 3, 1, 2)
        x_spatial = self.proj_spatial_detail(x_spatial)
        x_spatial = x_spatial.reshape(b, c, -1).permute(0, 2, 1)
        x_spatial = self.norm_detail_spatial_post(x_spatial)

        x_freq = self.cross_fusion_detail_freq(q_in=x_spatial, kv_in=x_freq, H=self.h, W=self.w)
        x_spatial = self.cross_fusion_detail_spatial(q_in=x_freq, kv_in=x_spatial, H=self.h, W=self.w)

        out = self.drop_detail(self.mlp_detail(torch.cat((x_spatial, x_freq), dim=-1)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def global_att(self, x):
        b, c, h, w = x.shape

        x_freq = self.sem_att_freq(x)
        x_freq = self.proj_freq_sem(x_freq)
        x_freq = x_freq.reshape(b, c, -1).permute(0, 2, 1)
        x_freq = self.norm_sem_freq_post(x_freq)

        x_spatial = x.reshape(b, c, -1).permute(0, 2, 1)
        x_spatial = self.norm_sem_spatial(x_spatial)
        x_spatial = self.sem_att_spatial(x_spatial, self.h, self.w)
        x_spatial = x_spatial.permute(0, 2, 1).reshape(b, c, h, w)
        x_spatial = self.proj_spatial_sem(x_spatial)
        x_spatial = x_spatial.reshape(b, c, -1).permute(0, 2, 1)
        x_spatial = self.norm_sem_spatial_post(x_spatial)

        x_freq = self.cross_fusion_sem_freq(q_in=x_spatial, kv_in=x_freq, H=self.h, W=self.w)
        x_spatial = self.cross_fusion_sem_spatial(q_in=x_freq, kv_in=x_spatial, H=self.h, W=self.w)

        out = self.drop_sem(self.mlp_sem(torch.cat((x_spatial, x_freq), dim=-1)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def global_att_spatial_only(self, x):
        b, c, h, w = x.shape

        x_spatial = x.reshape(b, c, -1).permute(0, 2, 1)
        x_spatial = self.norm_sem_spatial(x_spatial)
        x_spatial = self.sem_att_spatial(x_spatial, self.h, self.w)
        x_spatial = x_spatial.permute(0, 2, 1).reshape(b, c, h, w)
        x_spatial = self.proj_spatial_sem(x_spatial)
        x_spatial = x_spatial.reshape(b, c, -1).permute(0, 2, 1)

        out = self.drop_sem(self.mlp_sem(self.norm_sem(x_spatial)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def global_att_freq_only(self, x):
        b, c, h, w = x.shape

        x_freq = self.norm_sem_freq(x.reshape(b, c, -1).permute(0, 2, 1))
        x_freq = x_freq.permute(0, 2, 1).reshape(b, c, h, w)
        x_freq = self.sem_att_freq(x_freq)
        x_freq = self.proj_freq_sem(x_freq)
        x_freq = x_freq.reshape(b, c, -1).permute(0, 2, 1)

        out = self.drop_sem(self.mlp_sem(self.norm_sem(x_freq)))
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    def forward(self, x_high, x_low):
        b, c, h, w = x_high.shape

        x_high = self.proj_cnn(x_high)
        x_low = self.proj_tf(x_low)

        xa = x_high + x_low
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x_high * wei + 2 * x_low * (1 - wei)
        return xo


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
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
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


@MODELS.register_module()
class TuningFusion(BaseModule):
    def __init__(self, cnn_channels, cnn_layer_num, tf_channels, tf_layer_num, out_channels, out_index=[0, 1, 2, 3],
                 cnn_bottleneck=None, tf_bottleneck=None, resolution=None, sr_ratio=4):
        super(TuningFusion, self).__init__()

        assert cnn_bottleneck is not None
        assert tf_bottleneck is not None
        self.out_index = out_index
        self.fusion = nn.ModuleList()
        self.cnn_detail_ext = nn.ModuleList()
        self.tf_detail_ext = nn.ModuleList()
        self.cnn_base_ext = nn.ModuleList()
        self.tf_base_ext = nn.ModuleList()
        for i in range(cnn_layer_num):
            self.fusion.append(AFF(cnn_channel=cnn_channels[i], tf_channel=tf_channels[i],
                                   channels=out_channels[i],
                                   h=resolution[i], w=resolution[i], sr_ratio=sr_ratio))

        self.cnn_adapter = nn.ModuleList()
        self.tf_adapter = nn.ModuleList()
        for i in range(cnn_layer_num):
            self.cnn_adapter.append(Adapter(d_model=cnn_channels[i], out_dim=cnn_channels[i],
                                            bottleneck=cnn_channels[i] // cnn_bottleneck,
                                            adapter_layernorm_option='out'))
        for i in range(tf_layer_num):
            self.tf_adapter.append(Adapter(d_model=tf_channels[i], out_dim=tf_channels[i],
                                           bottleneck=tf_channels[i] // tf_bottleneck, adapter_layernorm_option='out'))

    def spatial_alignment(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):
        cnn_feat_from_list = []
        cnn_feat_to_list = []
        tf_feat_from_list = []
        tf_feat_to_list = []

        for idx, (cnn_m, tf_m) in enumerate(zip(self.cnn_adapter, self.tf_adapter)):
            cnn_feat_from_list.append(cnn_m(cnn_feat_from[idx]))
            cnn_feat_to_list.append(cnn_m(cnn_feat_to[idx]))

            if tf_feat_from[idx].shape[-2:] != cnn_feat_from[idx].shape[-2:]:
                tf_feat_from_resize = F.interpolate(tf_feat_from[idx], size=cnn_feat_from[idx].shape[-2:])
                tf_feat_to_resize = F.interpolate(tf_feat_to[idx], size=cnn_feat_from[idx].shape[-2:])
                tf_feat_from_list.append(tf_m(tf_feat_from_resize))
                tf_feat_to_list.append(tf_m(tf_feat_to_resize))

            else:
                tf_feat_from_list.append(tf_m(tf_feat_from[idx]))
                tf_feat_to_list.append(tf_m(tf_feat_to[idx]))

        return cnn_feat_from_list, tf_feat_from_list, cnn_feat_to_list, tf_feat_to_list

    def forward(self, cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to):

        cnn_feat_from_list, tf_feat_from_list, cnn_feat_to_list, tf_feat_to_list = \
            self.spatial_alignment(cnn_feat_from, tf_feat_from, cnn_feat_to, tf_feat_to)

        feats_from = [self.fusion[idx](cnn_f, tf_f) for idx, (cnn_f, tf_f) in
                      enumerate(zip(cnn_feat_from_list, tf_feat_from_list))]
        feats_to = [self.fusion[idx](cnn_f, tf_f) for idx, (cnn_f, tf_f) in
                    enumerate(zip(cnn_feat_to_list, tf_feat_to_list))]

        out_from = []
        out_to = []
        for idx in self.out_index:
            out_from.append(feats_from[idx])
            out_to.append(feats_to[idx])
        return out_from, out_to

