# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[20, 25], min_lr=1e-6, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, interval=1000)
# metric='all' 才能拿到 Fscore/Precision/Recall 的逐类指标
evaluation = dict(interval=1, metric='all', pre_eval=True)