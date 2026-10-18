"""数据集发现与解析：把「数据集名」映射到根目录。

约定：每个数据集是 <data_base>/<名称>/ 且含标准结构::

    <名称>/
      train/{A,B,label}
      val/{A,B,label}
      test/{A,B,label}

只要存在 train/A 即视为一个数据集。这样新增数据集 = 往 data/ 下放一个新文件夹，
无需改任何代码。
"""

import os

STRUCT_SPLITS = ("train", "val", "test")
STRUCT_SUBDIRS = ("A", "B", "label")


def is_dataset_dir(path: str) -> bool:
    """判断是否为标准数据集根目录（至少要有 train/A 与 train/label）。"""
    return (os.path.isdir(os.path.join(path, "train", "A"))
            and os.path.isdir(os.path.join(path, "train", "label")))


def discover_datasets(data_base: str = "data") -> dict:
    """扫描 data_base 下的所有数据集，返回 {名称: 根目录}（按名称排序）。"""
    found = {}
    if os.path.isdir(data_base):
        for name in sorted(os.listdir(data_base)):
            p = os.path.join(data_base, name)
            if os.path.isdir(p) and is_dataset_dir(p):
                found[name] = p
    return found


def resolve_root(data_base: str = "data", dataset: str = None, root: str = None):
    """确定数据根目录，返回 (root, dataset_name)。

    优先级：显式 root > data_base/dataset > data_base 本身。
    dataset 名为空时会用 root 的末级目录名补齐，便于输出命名。
    """
    if root:
        return root, (dataset or os.path.basename(os.path.normpath(root)))
    if dataset:
        return os.path.join(data_base, dataset), dataset
    return data_base, ""


def format_dataset_table(data_base: str = "data") -> str:
    found = discover_datasets(data_base)
    if not found:
        return f"在 {data_base} 下未发现标准数据集（需含 train/A、train/label）"
    lines = [f"在 {data_base} 下发现 {len(found)} 个数据集："]
    for name, path in found.items():
        lines.append(f"  - {name:<16} -> {path}")
    return "\n".join(lines)
