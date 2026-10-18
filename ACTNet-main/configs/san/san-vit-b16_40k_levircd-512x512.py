_base_ = [
    '../_base_/models/san_vit-b16.py', '../common/standard_512x512_40k_levircd.py',
    # '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# By default, models are trained on 4 GPUs with 8 images per GPU
train_dataloader = dict(batch_size=8, num_workers=4)

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  # noqa

model = dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name='levir_cd'),
    decode_head=dict(
        san_cfg=dict(
            in_channels=3),
        maskgen_cfg=dict(
            sos_token_format='learnable_token',
        ),
        # neck_cfg=dict(
        #     type='DifferentialMap',
        #     in_channels=[480, 480],
        #     out_channels=[480, 480],
        #     num_outs=2,
        # ),
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]
