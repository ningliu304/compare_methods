# ACTNet 统一对比实验配置（双时相 RGB，train/val/test/{A,B,label}）
#
# 启动（在 ACTNet-main 目录下）:
#   python train.py configs/actnet/actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_100e_bcd.py \
#       --work-dir work_dirs/bcd
#
# 说明:
# - 基于原 msk_head 512 levir 配置，改为 100 epoch、固定 seed、指向本实验数据。
# - 原配置中 model type 笔误 'CTDIEncoderDecoder' 已修正为 'ACTDIEncoderDecoder'。
# - 预训练权重路径（internimage_xl / eva02-large / mit_b2）仍需按你的本地路径修改。
# - 依赖 DCNv3 + DCNv4 自定义算子编译（见 requirements/ 与 ops_dcnv3、DCNv4_op）。
# - 最优权重由 BestF1CheckpointHook 按「验证集变化类 F1」覆盖保存。
_base_ = ['./actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_80k_levir_cd.py']

# ================== 数据（统一结构） ==================
# 可通过环境变量 BCD_DATA_ROOT 切换数据集，无需改动本文件；默认指向 ../data
import os
data_root = os.environ.get('BCD_DATA_ROOT', '../data')
out_dir = os.environ.get('BCD_OUT_DIR', './runs/actnet')
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
    batch_size=2,
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
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True,
         begin=0, end=5, convert_to_iter_based=True),
    dict(type='PolyLR', power=0.9, begin=5, end=max_epochs,
         eta_min=0.0, by_epoch=True, convert_to_iter_based=True),
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

# 修正原配置笔误的模型类名
model = dict(type='ACTDIEncoderDecoder')

# ================== 随机种子（统一） ==================
randomness = dict(seed=42, deterministic=True)
