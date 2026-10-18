"""CMCDNet 统一评估入口（复用 common 的指标与可视化，保证与其他模型口径一致）。

CMCDNet 训练仍走 mmsegmentation 原生流程:
    python tools/train.py my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py --seed 42 --deterministic --work-dir work_dirs/cmcd
训练完成后用本脚本评估:
    # 传单个权重
    python test_bcd.py --root ../data --checkpoint work_dirs/cmcd/latest.pth --out-dir runs/cmcd
    # 或传目录：自动在验证集上按「变化类 F1」挑选最优权重，再在测试集报告
    python test_bcd.py --root ../data --checkpoint work_dirs/cmcd --out-dir runs/cmcd

注意: 需要本机可导入 mmseg/mmcv（CMCDNet 依赖 mmcv-full 1.3~1.4 自定义编译）。
"""

import os
import sys
import shutil
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)   # common
sys.path.insert(0, HERE)

from common.seed import set_seed
from common.dataset import CDDataset
from common.metrics import CDMetricTracker
from common.vis import visualize_batch
from common.checkpoint import resolve_checkpoints, load_weights
from common.datasets import resolve_root
from common.mmseg_log import append_test_result, artifact_paths


def build_loader(root, split, img_size, num_workers, norm):
    ds = CDDataset(root, split, img_size, augment=False, norm=norm)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)


def evaluate(model, loader, device, num_vis=0, vis_dir=None):
    """在 loader 上评估，返回 common 口径的指标 dict。"""
    tracker = CDMetricTracker(n_class=2)
    saved = 0
    with torch.no_grad():
        for batch in loader:
            a = batch["a"].to(device).float()
            b = batch["b"].to(device).float()
            gt = batch["label"].numpy()
            # CMCDNet 训练用 cv2 读取（BGR），此处 RGB->BGR 保持一致
            x = torch.cat([a.flip(1), b.flip(1)], dim=1)

            # 直接走 encode_decode：绕开 mmseg forward 对 img_metas 的强依赖，
            # 返回 [B, num_classes, H, W] logits（已 resize 回输入尺寸）
            out = model.encode_decode(x, img_metas=None)
            if isinstance(out, (list, tuple)):
                out = torch.cat(out, dim=0)
            pred = out.argmax(dim=1).long().cpu().numpy()    # [B,H,W]

            tracker.update(pred, gt)
            if vis_dir is not None and saved < num_vis:
                names = batch.get("name", ["0"])
                names = names if isinstance(names, (list, tuple)) else list(names)
                saved = visualize_batch(batch["a"], batch["b"], gt, pred, vis_dir,
                                        names, saved, num_vis, [0, 0, 0], [1, 1, 1])
    return tracker.scores()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data-base", default="data", help="数据集根目录")
    ap.add_argument("--dataset", default="", help="数据集名，如 Evlab / HRSCD / WUSU512")
    ap.add_argument("--root", default="", help="直接指定数据根目录")
    ap.add_argument("--out-dir", default="runs/cmcd")
    ap.add_argument("--config", default="my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py")
    ap.add_argument("--checkpoint", default="work_dirs/cmcd", help="单个 .pth 或权重目录")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-vis", type=int, default=8)
    args, _ = ap.parse_known_args()

    root, dataset = resolve_root(args.data_base, args.dataset, args.root)
    best_out, log_path, vis_dir = artifact_paths(args.out_dir, dataset)

    set_seed(42)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from mmseg.apis import init_segmentor
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "无法导入 mmseg（CMCDNet 依赖 mmcv-full 1.3~1.4 自定义编译）。\n"
            f"原始错误: {e}"
        )

    model = init_segmentor(args.config, None, device=device)
    model.eval()
    norm = "zero_one"   # 与原 wc_Normalize（仅 /255）一致

    if os.path.isfile(args.checkpoint):
        chosen = args.checkpoint
    elif os.path.isfile(best_out):
        # 训练时已按验证集变化类 F1 选好最优权重
        chosen = best_out
        print(f"[select] use training best -> {best_out}")
    else:
        ckpts = resolve_checkpoints(args.checkpoint)
        # 兼容旧产物：多候选权重在验证集上按变化类 F1 选最优
        if len(ckpts) > 1:
            val_loader = build_loader(root, "val", args.img_size,
                                      args.num_workers, norm)
            best_ckpt, best_f1 = None, -1.0
            for ckpt in ckpts:
                load_weights(model, ckpt, device)
                f1 = evaluate(model, val_loader, device)["f1"]
                print(f"[select] {os.path.basename(ckpt)}  val F1={f1:.4f}")
                if f1 > best_f1:
                    best_ckpt, best_f1 = ckpt, f1
            print(f"[select] best = {best_ckpt} (val F1={best_f1:.4f})")
            chosen = best_ckpt
        else:
            chosen = ckpts[0]

    if os.path.abspath(chosen) != os.path.abspath(best_out):
        shutil.copyfile(chosen, best_out)
        print(f"[save] selected best weight -> {best_out}")

    load_weights(model, chosen, device)
    test_loader = build_loader(root, "test", args.img_size, args.num_workers, norm)
    s = evaluate(model, test_loader, device, num_vis=args.num_vis, vis_dir=vis_dir)

    tag = f" @ {dataset}" if dataset else ""
    print("=" * 60)
    print(f"[TEST] CMCDNet{tag}  (ckpt: {os.path.basename(chosen)})")
    print(f"  Precision={s['precision']:.4f}  Recall={s['recall']:.4f}  "
          f"F1={s['f1']:.4f}  IoU={s['iou']:.4f}")
    print(f"  mIoU={s['miou']:.4f}  mF1={s['mf1']:.4f}  OA={s['acc']:.4f}")
    print(f"  visualizations -> {vis_dir}")
    print("=" * 60)
    append_test_result(log_path, f"CMCDNet{tag}", s, vis_dir)


if __name__ == "__main__":
    main()
