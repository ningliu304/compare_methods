import math
import os
import sys

import torch
from mmcv.runner.hooks import HOOKS, Hook, LrUpdaterHook

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.mmseg_log import (append_epoch_header,  # noqa: E402
                              append_epoch_row, artifact_paths, to_float)


@HOOKS.register_module()
class ReduceOnPlateauLrUpdaterHook(LrUpdaterHook):

    """

    Must be used together with EvalHook or DistEvalHook with save_best not being None.
    EvalHook saves best checkpoint's evaluation score and file path
    in ``runner.meta['hook_msgs']`` when the save_best is set.

    """

    def __init__(self, factor=0.5, min_lr=5e-7, patience=3, **kwargs):
        self.factor = factor
        self.num_updates = 0
        self.min_lr = min_lr
        self.patience = patience
        self.no_optim = 0
        self.score = None
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if runner.meta is None:
            return base_lr

        msgs = runner.meta.get("hook_msgs", None)
        if msgs is None:
            return base_lr

        best_score = msgs.get("best_score", None)
        best_ckpt = msgs.get("best_ckpt", None)

        if best_score is None or best_ckpt is None:
            pass
        elif self.score is None:
            self.score = best_score
        elif self.score == best_score:
            self.no_optim += 1
            if self.no_optim > self.patience:
                runner.load_checkpoint(best_ckpt)
                self.no_optim = 0
                self.num_updates += 1
                runner.logger.info(
                    f'Reduce lr by a factor of {self.factor} and load model from the last best checkpoint.')
        else:
            self.score = best_score
            self.no_optim = 0

        lr = base_lr * math.pow(self.factor, self.num_updates)
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)

        return lr


@HOOKS.register_module()
class BestF1SaverHook(Hook):
    """按验证集「变化类 F1」覆盖保存唯一最优权重。

    需注册在 EvalHook（LOW=70）之后、LoggerHook（VERY_LOW=90）之前，
    例如 priority=80；从 log_buffer 读取本轮验证指标，同时按统一 Trainer
    的格式写 log_<数据集>.txt。
    """

    def __init__(self, out_dir, dataset='', class_name='change',
                 metric_key=None):
        self.out_dir = out_dir
        self.dataset = dataset
        self.class_name = class_name
        self.metric_key = metric_key or f'Fscore.{class_name}'
        self.best_f1 = -float('inf')
        self._loss_sum = 0.0
        self._loss_count = 0

    def before_run(self, runner):
        _, log_path, _ = artifact_paths(self.out_dir, self.dataset)
        append_epoch_header(log_path)

    def before_train_epoch(self, runner):
        self._loss_sum = 0.0
        self._loss_count = 0

    def after_train_iter(self, runner):
        outputs = getattr(runner, 'outputs', None) or {}
        log_vars = outputs.get('log_vars', {}) if isinstance(outputs, dict) else {}
        loss = to_float(log_vars.get('loss'))
        if loss is not None:
            self._loss_sum += loss
            self._loss_count += 1

    def after_train_epoch(self, runner):
        if runner.rank != 0:
            return
        output = getattr(runner.log_buffer, 'output', {}) or {}
        f1 = to_float(output.get(self.metric_key))
        if f1 is None:
            runner.logger.warning(
                f'{self.metric_key} not found in validation results; '
                'skip saving the best checkpoint this epoch.')
            return

        nan = float('nan')
        precision = to_float(output.get(f'Precision.{self.class_name}'), nan)
        recall = to_float(output.get(f'Recall.{self.class_name}'), nan)
        iou = to_float(output.get(f'IoU.{self.class_name}'), nan)
        train_loss = (self._loss_sum / self._loss_count
                      if self._loss_count else nan)

        best_path, log_path, _ = artifact_paths(self.out_dir, self.dataset)
        append_epoch_row(log_path, runner.epoch + 1, train_loss, nan,
                         precision, recall, f1, iou)

        if f1 > self.best_f1:
            self.best_f1 = f1
            model = getattr(runner.model, 'module', runner.model)
            os.makedirs(os.path.dirname(os.path.abspath(best_path)),
                        exist_ok=True)
            torch.save(model.state_dict(), best_path)
            runner.logger.info(
                f'best model updated (val change F1={f1:.4f}) -> {best_path}')
        self._loss_sum = 0.0
        self._loss_count = 0


