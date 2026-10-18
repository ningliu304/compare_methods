"""统一配置与命令行参数。所有模型的 train_cd.py 共用 parse_args()。"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List

from common.datasets import resolve_root, format_dataset_table, is_dataset_dir, discover_datasets

# 归一化预设：返回 (mean, std)
NORM_PRESETS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "minus_one_one": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "zero_one": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}


def resolve_norm(norm: str):
    if norm not in NORM_PRESETS:
        raise ValueError(f"未知的 norm 预设: {norm}，可选 {list(NORM_PRESETS)}")
    mean, std = NORM_PRESETS[norm]
    return list(mean), list(std)


@dataclass
class CDConfig:
    # 数据
    root: str = "data"
    dataset: str = ""             # 数据集名（用于给权重/日志/可视化命名）
    data_base: str = "data"       # 数据集根目录（其下每个子目录是一个数据集）
    img_size: int = 512
    num_workers: int = 4
    norm: str = "imagenet"

    # 训练
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    optim: str = "adamw"          # adam | adamw | sgd
    scheduler: str = "cosine"     # cosine | step | poly | linear | none
    seed: int = 42
    n_class: int = 2
    gpu: str = "0"

    # 模型（各 adapter 使用的网络名，如 ChangeFormerV6 / base_transformer_pos_s4_dd8）
    model: str = ""
    # 附加超参（adapter 可读，如 DSAMNet 阈值）
    extra: dict = field(default_factory=dict)

    # 输出
    out_dir: str = "runs/exp"
    num_vis: int = 8              # 测试集可视化样本数
    max_batches: int = 0          # >0 时每个 split 每轮最多跑这么多 batch（冒烟测试用）

    @property
    def device(self) -> str:
        import torch

        return f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"

    @property
    def norm_wh(self):
        return resolve_norm(self.norm)

    # ---- 由数据集名派生的产物命名（后缀 _<dataset>） ----
    @property
    def _suffix(self) -> str:
        return f"_{self.dataset}" if self.dataset else ""

    @property
    def ckpt_name(self) -> str:
        return f"best{self._suffix}.pth"

    @property
    def log_name(self) -> str:
        return f"log{self._suffix}.txt"

    @property
    def vis_name(self) -> str:
        return f"vis{self._suffix}"


def parse_args(argv: List[str] = None, defaults: dict = None) -> CDConfig:
    """解析统一 CLI。

    defaults: 模型专属默认超参（如 {'lr':1e-2,'optim':'sgd'}），会被写入 argparse
    默认值；用户命令行仍可覆盖。

    数据选择优先级：--root > (--data-base + --dataset) > --data-base。
    推荐用 `--dataset <名称>` 直接切换 <data-base>/<名称>/ 下的数据集。
    """
    p = argparse.ArgumentParser("统一变化检测对比实验")
    p.add_argument("--data-base", default="data", help="数据集根目录（其下每个子目录为一个数据集）")
    p.add_argument("--dataset", "--data", dest="dataset", default="",
                   help="数据集名，如 Evlab / HRSCD / WUSU512（--data 为别名）")
    p.add_argument("--root", default="", help="直接指定数据根目录（含 train/val/test/{A,B,label}）")
    p.add_argument("--list-datasets", action="store_true", help="列出 --data-base 下的可用数据集并退出")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--norm", default="imagenet", choices=list(NORM_PRESETS))

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optim", default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "poly", "linear", "none"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-class", type=int, default=2)
    p.add_argument("--gpu", default="0")

    p.add_argument("--model", default="", help="网络名（可选，adapter 默认值优先）")

    p.add_argument("--out-dir", default="runs/exp")
    p.add_argument("--num-vis", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0,
                   help=">0 时每个 split 每轮最多跑这么多 batch（冒烟测试用）")

    if defaults:
        p.set_defaults(**defaults)

    args = p.parse_args(argv)

    if args.list_datasets:
        print(format_dataset_table(args.data_base))
        sys.exit(0)

    root, dataset = resolve_root(args.data_base, args.dataset, args.root)

    # 快速失败：若 root 不是单个数据集，而是含多个数据集的根目录，给出可操作提示
    if not args.list_datasets and not is_dataset_dir(root):
        found = discover_datasets(root)
        if found:
            names = ", ".join(found)
            print(
                f"[错误] {root} 是「数据集根目录」，其下包含多个数据集，不能直接当数据集用。\n"
                f"       请改用 --dataset <名称>，例如: --dataset {next(iter(found))}\n"
                f"       可选数据集: {names}"
            )
            sys.exit(1)

    cfg = CDConfig(
        root=root,
        dataset=dataset,
        data_base=args.data_base,
        img_size=args.img_size,
        num_workers=args.num_workers,
        norm=args.norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optim=args.optim,
        scheduler=args.scheduler,
        seed=args.seed,
        n_class=args.n_class,
        gpu=args.gpu,
        model=args.model,
        out_dir=args.out_dir,
        num_vis=args.num_vis,
        max_batches=args.max_batches,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    return cfg
