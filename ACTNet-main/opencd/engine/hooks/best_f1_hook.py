import os
import sys

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.mmseg_log import (append_epoch_header,  # noqa: E402
                              append_epoch_row, artifact_paths, to_float)


@HOOKS.register_module()
class BestF1CheckpointHook(Hook):
    """按验证集「变化类 F1」覆盖保存唯一最优权重。

    同时按统一 Trainer 的格式写 log_<数据集>.txt。
    """

    def __init__(self, out_dir, dataset='', class_name='changed',
                 metric_key=None):
        self.out_dir = out_dir
        self.dataset = dataset
        self.class_name = class_name
        self.metric_key = metric_key or f'Fscore.{class_name}'
        self.best_f1 = -float('inf')
        self._loss_sum = 0.0
        self._loss_count = 0
        self._epoch_loss = float('nan')

    def before_run(self, runner):
        _, log_path, _ = artifact_paths(self.out_dir, self.dataset)
        append_epoch_header(log_path)

    def before_train_epoch(self, runner):
        self._loss_sum = 0.0
        self._loss_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        loss = to_float((outputs or {}).get('loss'))
        if loss is not None:
            self._loss_sum += loss
            self._loss_count += 1

    def after_train_epoch(self, runner):
        if self._loss_count:
            self._epoch_loss = self._loss_sum / self._loss_count
        self._loss_sum = 0.0
        self._loss_count = 0

    def _metric(self, metrics, key):
        value = metrics.get(key)
        if value is None:
            value = metrics.get(f'val/{key}')
        value = to_float(value)
        # mmseg IoUMetric 返回百分比（0~100），统一转成 Trainer 的 0~1
        return None if value is None else value / 100.0

    def after_val_epoch(self, runner, metrics=None):
        if runner.rank != 0:
            return
        metrics = metrics or {}
        f1 = self._metric(metrics, self.metric_key)
        if f1 is None:
            runner.logger.warning(
                f'{self.metric_key} not found in validation results; '
                'skip saving the best checkpoint this epoch.')
            return

        nan = float('nan')
        precision = self._metric(metrics, f'Precision.{self.class_name}')
        recall = self._metric(metrics, f'Recall.{self.class_name}')
        iou = self._metric(metrics, f'IoU.{self.class_name}')
        precision = nan if precision is None else precision
        recall = nan if recall is None else recall
        iou = nan if iou is None else iou

        best_path, log_path, _ = artifact_paths(self.out_dir, self.dataset)
        append_epoch_row(log_path, runner.epoch, self._epoch_loss, nan,
                         precision, recall, f1, iou)

        if f1 > self.best_f1:
            self.best_f1 = f1
            model = getattr(runner.model, 'module', runner.model)
            os.makedirs(os.path.dirname(os.path.abspath(best_path)),
                        exist_ok=True)
            torch.save(model.state_dict(), best_path)
            runner.logger.info(
                f'best model updated (val change F1={f1:.4f}) -> {best_path}')
