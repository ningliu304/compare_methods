# CMCDNet 双时相 RGB 变化检测配置（统一对比实验）
#
# 启动:
#   python tools/train.py my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py --seed 42 --deterministic
#
# 说明:
# - A 时相 -> opt 流(resnet50, 3通道)，B 时相 -> sar 流(resnet18, 3通道)。
# - 通道堆叠后 6 通道，CMCD backbone 按 (opt_channels=3, sar_channels=3) 切分。
import os

_base_ = [
    './base/bcd_rgb_dataset.py',
    './base/scheduler_100e.py',
    './base/default_runtime.py'
]

data_root = os.environ.get('BCD_DATA_ROOT', '../data')
out_dir = os.environ.get('BCD_OUT_DIR', '../runs/cmcd')
dataset = os.environ.get('BCD_DATASET') or os.path.basename(
    os.path.normpath(data_root))

# 最优权重由 BestF1SaverHook 按验证集变化类 F1 覆盖保存，不再按间隔存权重
checkpoint_config = dict(by_epoch=True, interval=1000)
custom_hooks = [
    dict(type='BestF1SaverHook', out_dir=out_dir, dataset=dataset,
         class_name='change', priority=80),
]

optimizer = dict(type='Adam', lr=2e-4)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CMCD',
        enc_opt_dims=[64, 256, 512, 1024, 2048],
        backbone_opt_cfg=dict(
            type='TIMMBackbone',
            model_name='resnet50',
            in_channels=3,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=32,
            pretrained=True),
        enc_sar_dims=[64, 64, 128, 256, 512],
        backbone_sar_cfg=dict(
            type='TIMMBackbone',
            model_name='resnet18',
            in_channels=3,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=32,
            pretrained=True),
        center_block="dblock",
        side_dim=64,
        norm_cfg=norm_cfg
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce',
                 loss_weight=1.0, class_weight=(1.0, 2.0)),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0,
                 class_weight=(1.0, 2.0))
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
