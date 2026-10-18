_base_ = [
    '../common/standard_512x512_40k_s2looking.py'
    # '../common/standard_256x256_100e_whucd.py'
]

# crop_size = (256, 256)
pretrained_cnn = '/home/debian/Workspace/change_adapter/pretained/internimage_xl_22k_192to384.pth'
pretrained_tf = '/home/debian/Workspace/change_adapter/pretained/' \
                'eva02-large-p14_m38m-pre_in21k-medft_3rdparty_in1k-448px_20230505-150dc5ed.pth'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[122.7709, 116.7460, 104.0937] * 2,
    std=[68.5005, 66.6322, 70.3232] * 2,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='ACTDIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    cnn_encoder_resolution=dict(
        size=(384, 384),
        mode='bilinear'),
    tf_encoder_resolution=dict(
        size=(448, 448),
        mode='bilinear'),
    cnn_cfg=dict(
        type='InternImage',
        core_op='DCNv3',
        num_stages=4,
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        frozen_exclude=['adapter'],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_cnn)),
    tf_cfg=dict(
        type='ViTEVA02',
        arch='l',
        img_size=448,
        out_indices=(1, 3, 21, 23),
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='featmap',
        frozen_stages=24,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_tf)
    ),
    tuning_fusing_cfg=dict(
        type='TuningFusion',
        cnn_channels=[192, 384, 768, 1536],
        cnn_layer_num=4,
        tf_channels=[1024, 1024, 1024, 1024],
        tf_layer_num=4,
        out_channels=[192, 384, 768, 1024],
        cnn_bottleneck=8,
        tf_bottleneck=8,
        out_index=[1, 2, 3],
        resolution=[96, 48, 24, 12]
    ),
    decode_head=dict(
        type='CNNTransformerAdapterHead',
        ban_cfg=dict(
            clip_channels=[192, 384, 768, 1024],
            fusion_index=[1, 2, 3],
            side_enc_cfg=dict(
                type='mmseg.MixVisionTransformer',
                init_cfg=dict(
                    type='Pretrained', checkpoint=checkpoint),
                in_channels=3,
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 4, 6, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1)),
        neck=dict(
            type='TTF',
            in_channels=[64, 128, 320, 512],
            out_channels=[64, 128, 320, 512],
            num_outs=4,
        ),
        ban_dec_cfg=dict(
            type='Mask2FormerHead',
            in_channels=[64, 128, 320, 512],
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            num_classes=num_classes,
            num_queries=10,
            num_transformer_feat_level=4,
            align_corners=False,
            pixel_decoder=dict(
                type='mmdet.MSDeformAttnPixelDecoder',
                num_outs=4,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(  # DeformableDetrTransformerEncoder
                    num_layers=6,
                    layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                        self_attn_cfg=dict(  # MultiScaleDeformableAttention
                            embed_dims=256,
                            num_heads=8,
                            num_levels=4,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=True,
                            norm_cfg=None,
                            init_cfg=None),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True))),
                    init_cfg=None),
                positional_encoding=dict(  # SinePositionalEncoding
                    num_feats=128, normalize=True),
                init_cfg=None),
            enforce_decoder_input_project=False,
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            transformer_decoder=dict(  # Mask2FormerTransformerDecoder
                return_intermediate=True,
                num_layers=12,
                layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                    cross_axial_attn=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0,
                    ),
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True)),
                init_cfg=None),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * num_classes + [0.1]),
            loss_mask=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type='mmdet.DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(
                            type='mmdet.CrossEntropyLossCost',
                            weight=5.0,
                            use_sigmoid=True),
                        dict(
                            type='mmdet.DiceCost',
                            weight=5.0,
                            pred_act=True,
                            eps=1.0)
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler')))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

max_iter = 80000
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=max_iter,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iter, val_interval=max_iter // 20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=max_iter // 20),
    # visualization=dict(type='CDVisualizationHook', interval=1)
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'cnn_cfg': dict(lr_mult=0.1, decay_mult=1.0),
            'tf_cfg': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    # loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

find_unused_parameters = True

train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=1)
