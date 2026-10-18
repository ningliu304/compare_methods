_base_ = [
    '../_base_/models/changeformer_mit-b0.py',
    '../common/standard_256x256_100e_whucd.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 5, 8]),
    decode_head=dict(in_channels=[v * 2 for v in [64, 128, 320, 512]], num_classes=2))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

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
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iter, val_interval=max_iter // 20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=max_iter // 20, save_best='mIoU'))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))