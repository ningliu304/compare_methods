"""FAEWNet 统一评估入口（复用 common 的指标与可视化，口径与其他模型一致）。

FAEWNet 训练走 OpenCD/MMSeg 原生入口（配置已设 100 epoch、seed=42）：
    BCD_DATA_ROOT=data/Evlab python tools/train.py \
        configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py --work-dir runs/faewnet_Evlab

训练完成后用统一口径评估并可视化（传目录时按验证集变化类 F1 自动挑选最优权重）：
    python FAEWNet-main/test_cd.py --dataset Evlab \
        --config FAEWNet-main/configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py \
        --checkpoint runs/faewnet_Evlab --out-dir runs/faewnet

依赖: mmseg 1.x / mmcv 2.x + mmpretrain + mmdet + peft。
若因环境 API 差异报错，可改用原生 `python tools/test.py <config> <ckpt>`。
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


def build_loader(root, split, img_size, num_workers):
    ds = CDDataset(root, split, img_size, augment=False, norm="zero_one")
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)


def evaluate(model, loader, device, SegDataSample, num_vis=0, vis_dir=None):
    tracker = CDMetricTracker(n_class=2)
    saved = 0
    with torch.no_grad():
        for batch in loader:
            a = batch["a"][0].float()      # [3,H,W] RGB, 0~1
            b = batch["b"][0].float()
            gt = batch["label"].numpy()    # [1,H,W]

            # 训练用 cv2 读入(BGR)后由 data_preprocessor 转 RGB，这里 feed BGR 保持一致
            x6 = torch.cat([a.flip(0), b.flip(0)], dim=0) * 255.0   # [6,H,W] BGR 0~255
            h, w = x6.shape[-2:]
            ds = SegDataSample()
            ds.set_metainfo(dict(
                ori_shape=(h, w, 3), img_shape=(h, w, 3), pad_shape=(h, w, 3),
                scale_factor=(1.0, 1.0), flip=False))

            result = model.test_step(dict(inputs=[x6], data_samples=[ds]))[0]
            pred = result.pred_sem_seg.data.squeeze(0).long().cpu().numpy()[None]

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
    ap.add_argument("--dataset", "--data", dest="dataset", default="", help="数据集名，如 Evlab / HRSCD / WUSU512")
    ap.add_argument("--root", default="", help="直接指定数据根目录")
    ap.add_argument("--out-dir", default="runs/faewnet")
    ap.add_argument("--config", default="configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py")
    ap.add_argument("--checkpoint", default="runs/faewnet", help="单个 .pth 或权重目录")
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
        from mmseg.apis import init_model
        from mmseg.structures import SegDataSample
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "无法导入 mmseg（FAEWNet 依赖 mmseg 1.x / mmcv 2.x 与 mmpretrain / mmdet / peft）。\n"
            f"原始错误: {e}"
        )

    model = init_model(args.config, None, device=device)
    model.eval()

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
                                      args.num_workers)
            best_ckpt, best_f1 = None, -1.0
            for ckpt in ckpts:
                load_weights(model, ckpt, device)
                f1 = evaluate(model, val_loader, device, SegDataSample)["f1"]
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
    test_loader = build_loader(root, "test", args.img_size, args.num_workers)
    s = evaluate(model, test_loader, device, SegDataSample,
                 num_vis=args.num_vis, vis_dir=vis_dir)

    tag = f" @ {dataset}" if dataset else ""
    print("=" * 60)
    print(f"[TEST] FAEWNet{tag}  (ckpt: {os.path.basename(chosen)})")
    print(f"  Precision={s['precision']:.4f}  Recall={s['recall']:.4f}  "
          f"F1={s['f1']:.4f}  IoU={s['iou']:.4f}")
    print(f"  mIoU={s['miou']:.4f}  mF1={s['mf1']:.4f}  OA={s['acc']:.4f}")
    print(f"  visualizations -> {vis_dir}")
    print("=" * 60)
    append_test_result(log_path, f"FAEWNet{tag}", s, vis_dir)


if __name__ == "__main__":
    main()
