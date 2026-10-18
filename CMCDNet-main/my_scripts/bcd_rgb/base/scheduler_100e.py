# 统一 100 epoch 训练调度
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1000)
# 指标：同时报告 mIoU / mFscore(F1) / Precision / Recall
# 不设 save_best；最优权重由 BestF1SaverHook 按「验证集变化类 F1」覆盖保存
evaluation = dict(interval=1, metric='all', pre_eval=True)
