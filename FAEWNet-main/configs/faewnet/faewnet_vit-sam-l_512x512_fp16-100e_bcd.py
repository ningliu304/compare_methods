# FAEWNet 统一对比实验配置（双时相 RGB，train/val/test/{A,B,label}，512×512，100 epoch）
#
# 启动（在 FAEWNet-main 目录下，或从根目录用相对路径）:
#   BCD_DATA_ROOT=data/Evlab python tools/train.py \
#       configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py --work-dir runs/faewnet_Evlab
#
# 说明:
# - 基于原 fp16-719e 配置，改为 100 epoch、固定 seed、数据指向本实验（用 BCD_DATA_ROOT 切换数据集）。
# - 需 mmseg>=1.0.0rc6、mmcv>=2.0.0、mmpretrain、mmdet、peft（SAM ViT-L 预训练权重自动下载）。
# - 不设 save_best（原为 mIoU）；最优权重由 BestF1CheckpointHook 按「验证集变化类 F1」覆盖保存。
_base_ = ['./faewnet_vit-sam-l_512x512_fp16-719e_levircd.py']

import os

# ================== 数据（统一结构） ==================
data_root = os.environ.get('BCD_DATA_ROOT', '../data')
out_dir = os.environ.get('BCD_OUT_DIR', './runs/faewnet')
dataset = os.environ.get('BCD_DATASET') or os.path.basename(
    os.path.normpath(data_root))

# 验证/测试输入统一为 512×512（与统一评估脚本口径一致）
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(512, 512), keep_ratio=False),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train/label',
            img_path_from='train/A',
            img_path_to='train/B')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A',
            img_path_to='val/B'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='test/label',
            img_path_from='test/A',
            img_path_to='test/B'),
        pipeline=test_pipeline))

# 补充逐类指标（Fscore.changed 等），用于按变化类 F1 选最优权重
val_evaluator = dict(type='opencd.CDIoUMetric',
                     iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator

# ================== 训练：100 epoch ==================
max_epochs = 100
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-4, by_epoch=True,
         begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=max_epochs, begin=5,
         by_epoch=True, end=max_epochs, convert_to_iter_based=True),
]
default_hooks = dict(
    # 不按间隔存权重；最优权重由 BestF1CheckpointHook 覆盖保存
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1000,
                    save_best=None),
    visualization=dict(type='CDVisualizationHook', interval=1,
                       img_shape=(512, 512, 3)),
)
custom_hooks = [
    dict(type='BestF1CheckpointHook', out_dir=out_dir, dataset=dataset,
         class_name='changed'),
]

# ================== 随机种子（统一） ==================
randomness = dict(seed=42, deterministic=True)
