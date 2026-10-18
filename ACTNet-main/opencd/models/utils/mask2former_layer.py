# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import math

from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.functional import linear, softmax
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn import functional as F

from mmengine.model import BaseModule

from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoder
from mmdet.models.layers.transformer.detr_layers import DetrTransformerDecoder, DetrTransformerDecoderLayer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention


class Mask2FormerTransformerEncoder(DeformableDetrTransformerEncoder):
    """Encoder in PixelDecoder of Mask2Former."""

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                reference_points: Tensor, **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim). If not None, it will be added to the
                `query` before forward function. Defaults to None.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query


class Mask2FormerTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            Mask2FormerTransformerDecoderLayer(i, **self.layer_cfg)
            for i in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]

class rcda_rebuild(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(5 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(5 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.lin = torch.nn.Linear(embed_dim, embed_dim)
        # self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # self.senet = SEBlock(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query_row, query_col, key_row, key_col, value):
        v_avg = self.sigmoid(self.lin(self.avg_pool(value.permute(0,3,1,2)).squeeze(-1).squeeze(-1))).unsqueeze(-1).permute(2,0,1)
        # v_avg = self.senet(value.permute(0,3,1,2)).squeeze(-1).permute(2,0,1)

        bsz, tgt_len, embed_dim = query_row.size()
        src_len_row = key_row.size()[2]
        src_len_col = key_col.size()[1]

        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_row = linear(query_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 1
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q_col = linear(query_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = embed_dim * 3
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_row = linear(key_row, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 3
        _end = embed_dim * 4
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k_col = linear(key_col, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 4
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = linear(value, _w, _b)

        q_row = q_row.transpose(0, 1)
        q_col = q_col.transpose(0, 1)
        k_row = k_row.mean(1).transpose(0, 1) * v_avg
        k_col = k_col.mean(2).transpose(0, 1) * v_avg

        q_row = q_row * scaling
        q_col = q_col * scaling

        q_row = q_row.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        q_col = q_col.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

        k_row = k_row.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k_col = k_col.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().permute(1,2,0,3).reshape(src_len_col,src_len_row, bsz*num_heads, head_dim).permute(2,0,1,3)

        attn_output_weights_row = torch.bmm(q_row, k_row.transpose(1, 2))
        attn_output_weights_col = torch.bmm(q_col, k_col.transpose(1, 2))

        attn_output_weights_col = softmax(attn_output_weights_col, dim=-1)
        attn_output_weights_row = softmax(attn_output_weights_row, dim=-1)

        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
        attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        ### the following code base on einsum get the same results
        # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
        # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)

        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, torch.einsum("bqw,bqh->qbhw",attn_output_weights_row,attn_output_weights_col).reshape(tgt_len,bsz,num_heads,src_len_col,src_len_row).mean(2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class CrossAttentionLayerRCDA_v5(nn.Module):

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = rcda_rebuild(embed_dims, num_heads)

        self.norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.adapt_pos1d = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pos2posemb1d(self, pos, num_pos_feats=256, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., None] / dim_t
        posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return posemb

    def with_query_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        tensor = tensor.permute(1, 0, 2)
        pos = pos.permute(1, 0, 2)
        pos = self.adapt_pos1d(self.pos2posemb1d(pos[..., ax]))
        return tensor + pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor], ax):
        if pos is None:
            return tensor
        b, e, h, w = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]
        tensor = tensor.permute(0, 2, 3, 1)
        pos = pos.permute(0, 2, 3, 1)
        if ax == 0:
            pos = pos.mean(1).unsqueeze(1).repeat(1, h, 1, 1)
        else:
            pos = pos.mean(2).unsqueeze(2).repeat(1, 1, w, 1)
        return tensor + pos

    # query[5,4,256] memory[4,256,128,128]
    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # We return the generated attention map
        tgt = tgt.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        tgt2, attn = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos, 0),
                                         self.with_query_pos_embed(tgt, query_pos, 1),
                                         self.with_pos_embed(memory, pos, 0),
                                         self.with_pos_embed(memory, pos, 1),
                                         memory.permute(0, 2, 3, 1))

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(self.with_query_pos_embed(tgt, query_pos, 0),
                                   self.with_query_pos_embed(tgt, query_pos, 1),
                                   self.with_pos_embed(memory, pos, 0),
                                   self.with_pos_embed(memory, pos, 1),
                                   memory.permute(0, 2, 3, 1))

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        b, n, c = memory.shape
        memory = memory.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
        pos = pos.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class Mask2FormerTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in Mask2Former transformer."""
    def __init__(self, layer_idx, cross_axial_attn,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None):
        super(Mask2FormerTransformerDecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.cross_axial_attn = cross_axial_attn

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
                    dimension of all DETRs in mmdet is `batch`, \
                    please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
                    dimension of all DETRs in mmdet is `batch`, \
                    please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        if self.layer_idx % 4 == 3:
            self.cross_attn = CrossAttentionLayerRCDA_v5(**self.cross_axial_attn)
        else:
            self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                layer_idx: int = None,
                **kwargs) -> Tensor:
        """
        Args:
            layer_idx:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        if layer_idx % 4 == 3:
            query, _ = self.cross_attn(
                tgt=query, memory=key,
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=key_pos, query_pos=query_pos
            )
            query = query.permute(1, 0, 2)
        else:
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        query = self.norms[0](query)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query
