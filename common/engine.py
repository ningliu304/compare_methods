"""统一训练引擎。

Trainer 负责：构建模型/优化器/调度器、100 轮 train+val 循环、按 val **F1（变化类）** 保存最优
checkpoint、测试集评估并打印最终 P/R/F1/IoU、导出可视化。模型差异全部由 adapter 提供。
"""

import os

import numpy as np
import torch

from common.metrics import CDMetricTracker
from common.vis import visualize_batch


def _make_optimizer(cfg, params):
    if cfg.optim == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    if cfg.optim == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_scheduler(cfg, optimizer):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 4), gamma=0.5)
    if cfg.scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: (1 - e / float(cfg.epochs)) ** 0.9
        )
    if cfg.scheduler == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: max(0.0, 1.0 - e / float(cfg.epochs))
        )
    return None


class Trainer:
    def __init__(self, cfg, loaders, adapter, logger=None):
        self.cfg = cfg
        self.loaders = loaders
        self.adapter = adapter
        self.logger = logger

        self.device = torch.device(cfg.device)
        self.model = adapter.build_model(cfg).to(self.device)
        self.optimizer = _make_optimizer(cfg, self.model.parameters())
        self.scheduler = _make_scheduler(cfg, self.optimizer)
        self.tracker = CDMetricTracker(n_class=cfg.n_class)

        self.best_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
        self.vis_dir = os.path.join(cfg.out_dir, cfg.vis_name)
        os.makedirs(cfg.out_dir, exist_ok=True)

        if self.logger:
            n_params = sum(p.numel() for p in self.model.parameters())
            tag = f" | dataset={cfg.dataset}" if cfg.dataset else ""
            self.logger.info(
                f"device={self.device} | model={adapter.name}{tag} | "
                f"root={cfg.root} | params={n_params/1e6:.2f}M")

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _run_epoch(self, loader, train: bool, collect_vis: bool = False):
        self.model.train(train)
        self.tracker.reset()
        total_loss, n_batches = 0.0, 0
        vis_state = {"idx": 0}
        iterator = loader
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
        except ImportError:
            pass

        for batch in iterator:
            inputs, target = self.adapter.prepare_batch(batch, self.device)
            gt = batch["label"].numpy()  # [B,H,W] 0/1 int，用于统一指标

            with torch.set_grad_enabled(train):
                out = self.adapter.forward(self.model, inputs)
                loss = self.adapter.compute_loss(self.model, out, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            pred = self.adapter.predict_binary(out, inputs)
            pred = pred.detach().cpu().numpy()
            self.tracker.update(pred, gt)
            total_loss += float(loss.detach())
            n_batches += 1

            if self.cfg.max_batches and n_batches >= self.cfg.max_batches:
                break

            if collect_vis and vis_state["idx"] < self.cfg.num_vis:
                names = batch.get("name", [f"{i}" for i in range(len(gt))])
                vis_state["idx"] = visualize_batch(
                    batch["a"], batch["b"], gt, pred, self.vis_dir,
                    names if isinstance(names, (list, tuple)) else list(names),
                    vis_state["idx"], self.cfg.num_vis,
                    *self.cfg.norm_wh,
                )

        mean_loss = total_loss / max(1, n_batches)
        return {"loss": mean_loss, **self.tracker.scores()}

    # ------------------------------------------------------------------ #
    def fit(self):
        from common.logger import EPOCH_HEADER, format_epoch_row

        self._log(EPOCH_HEADER)
        best_f1 = -1.0
        for epoch in range(1, self.cfg.epochs + 1):
            tr = self._run_epoch(self.loaders["train"], train=True)
            va = self._run_epoch(self.loaders["val"], train=False)
            if self.scheduler is not None:
                self.scheduler.step()

            self._log(format_epoch_row(epoch, tr["loss"], va, val_loss=va["loss"]))

            # 最优模型按验证集变化类 F1 选择（非均值 mF1）
            if va["f1"] > best_f1:
                best_f1 = va["f1"]
                torch.save(self.model.state_dict(), self.best_path)
                self._log(f"  * best model updated (val F1={best_f1:.4f})")

        self._log(f"Done. best val F1={best_f1:.4f}, checkpoint={self.best_path}")
        return self.test()

    # ------------------------------------------------------------------ #
    def test(self):
        if os.path.exists(self.best_path):
            state = torch.load(self.best_path, map_location=self.device)
            self.model.load_state_dict(state)
            self._log(f"loaded best checkpoint: {self.best_path}")

        res = self._run_epoch(self.loaders["test"], train=False, collect_vis=True)
        self._log("=" * 60)
        _tag = f" @ {self.cfg.dataset}" if self.cfg.dataset else ""
        self._log(f"[TEST] {self.adapter.name}{_tag}")
        self._log(
            "  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  IoU={iou:.4f}".format(**res)
        )
        self._log(
            "  mIoU={miou:.4f}  mF1={mf1:.4f}  OA={acc:.4f}".format(**res)
        )
        self._log(f"  visualizations -> {self.vis_dir}")
        self._log("=" * 60)
        return res
