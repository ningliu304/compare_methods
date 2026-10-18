# 双时相 RGB 变化检测数据集配置（适配 train/val/test/{A,B,label}）
#
# A 作为 "opt" 流(3通道)，B 作为 "sar" 流(3通道)，通道堆叠后为 6 通道，
# CMCD backbone 按 (opt_channels, sar_channels) 切分（见 cmcd.py）。
dataset_type = 'WCDataset'
# 可通过环境变量 BCD_DATA_ROOT 切换数据集，无需改动本文件；默认指向 ../data
import os
data_root = os.environ.get('BCD_DATA_ROOT', '../data')
classes = ("bg", "change")
palette = ((0, 0, 0), (255, 0, 0))

img_scale = (512, 512)

train_pipeline = [
    dict(type='wc_LoadImageFromFile'),
    dict(type='wc_StackByChannel', keys=('img', 'aux')),
    dict(type='LoadAnnotations', label_remap={255: 1}),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='wc_Normalize'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='wc_LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='wc_StackByChannel', keys=('img', 'aux')),
            dict(type='wc_Normalize'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/A',
        aux_dir='train/B',
        ann_dir='train/label',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
        # 标注为 0/255，重映射为 0/1 供评估；避免 255 命中 ignore_index
        gt_seg_map_loader_cfg=dict(label_remap={255: 1})
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/A',
        aux_dir='val/B',
        ann_dir='val/label',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        gt_seg_map_loader_cfg=dict(label_remap={255: 1})
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/A',
        aux_dir='test/B',
        ann_dir='test/label',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        gt_seg_map_loader_cfg=dict(label_remap={255: 1})
    ))
