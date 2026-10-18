# Copyright (c) Open-CD. All rights reserved.
from typing import Dict

import numpy as np

from mmseg.evaluation import IoUMetric
from opencd.registry import METRICS


@METRICS.register_module()
class CDIoUMetric(IoUMetric):
    """在 IoUMetric 汇总指标之外补充逐类指标（如 Fscore.changed）。

    数值沿用 mmseg 的百分比口径（0~100），便于 CheckpointHook/自定义 Hook
    统一按变化类 F1 选最优。
    """

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = super().compute_metrics(results)

        areas = tuple(zip(*results))
        if len(areas) != 4:
            return metrics
        ret_metrics = self.total_area_to_metrics(
            sum(areas[0]), sum(areas[1]), sum(areas[2]), sum(areas[3]),
            self.metrics, self.nan_to_num, self.beta)
        class_names = self.dataset_meta['classes']
        for key, values in ret_metrics.items():
            if key == 'aAcc':
                continue
            for idx, name in enumerate(class_names):
                metrics[f'{key}.{name}'] = float(
                    np.round(float(values[idx]) * 100, 2))
        return metrics
