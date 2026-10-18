#!/usr/bin/env bash
# ============================================================
#  一键跑「所有模型 × 所有数据集」
#
#  用法（在 E:/论文/对比实验 根目录下）:
#    bash run_all.sh                       # 8 个 Trainer 模型 × 全部数据集
#    DATASETS="Evlab" bash run_all.sh      # 只跑一个数据集
#    EPOCHS=1 bash run_all.sh              # 冒烟测试（每个只跑 1 轮）
#    WITH_MMSEG=1 bash run_all.sh          # 额外跑 CMCDNet / ACTNet（需 mmcv 环境）
#    UNIFIED="--norm imagenet --lr 1e-4 --optim adamw" bash run_all.sh  # 强制统一超参
#
#  产物命名：权重/日志/可视化都会带上数据集名，例如
#    runs/bit_cd/best_Evlab.pth  runs/bit_cd/log_Evlab.txt  runs/bit_cd/vis_Evlab/
# ============================================================
set -u

DATA_BASE=${DATA_BASE:-data}
DATASETS=${DATASETS:-"Evlab HRSCD WUSU512"}
OUT_BASE=${OUT_BASE:-runs}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}
UNIFIED=${UNIFIED:-}          # 额外统一参数（可选）

# 8 个统一 Trainer 模型: "目录:标签"
TRAINER_MODELS=(
  "BIT_CD-master:bit_cd"
  "ChangeFormer-main:changeformer"
  "Tiny_model_4_CD-main:tinycd"
  "Siam-NestedUNet-master:snunet"
  "DSAMNet-main:dsamnet"
  "BCE-Net-main:bcenet"
  "ChangeViT-master:changevit"
  "ChangeMamba-master:changemamba"
)

echo "============================================================"
echo "数据集: $DATASETS"
echo "输出根目录: $OUT_BASE    epoch: $EPOCHS    seed: $SEED"
echo "============================================================"

for DS in $DATASETS; do
  echo
  echo "############ 数据集: $DS ############"

  # ---- 1) 统一 Trainer 模型 ----
  for M in "${TRAINER_MODELS[@]}"; do
    DIR="${M%%:*}"; TAG="${M##*:}"
    echo ">>> [$DS] $TAG"
    python "$DIR/train_cd.py" \
      --data-base "$DATA_BASE" --dataset "$DS" \
      --out-dir "$OUT_BASE/$TAG" \
      --epochs "$EPOCHS" --seed "$SEED" $UNIFIED || {
        echo "!!! [$DS] $TAG 失败，继续下一个"; }
  done

  # ---- 2) MMSeg 系: CMCDNet / ACTNet / FAEWNet（需匹配的 mmcv 环境） ----
  if [ "${WITH_MMSEG:-0}" = "1" ]; then
    echo ">>> [$DS] CMCDNet"
    BCD_DATA_ROOT="$DATA_BASE/$DS" BCD_DATASET="$DS" \
      BCD_OUT_DIR="$OUT_BASE/cmcd" python CMCDNet-main/tools/train.py \
      CMCDNet-main/my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py \
      --seed "$SEED" --deterministic --work-dir "$OUT_BASE/cmcd_$DS" || echo "!!! CMCDNet 训练失败"
    python CMCDNet-main/test_bcd.py \
      --data-base "$DATA_BASE" --dataset "$DS" \
      --config CMCDNet-main/my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py \
      --checkpoint "$OUT_BASE/cmcd_$DS" --out-dir "$OUT_BASE/cmcd" || echo "!!! CMCDNet 评估失败"

    echo ">>> [$DS] ACTNet"
    BCD_DATA_ROOT="$DATA_BASE/$DS" BCD_DATASET="$DS" \
      BCD_OUT_DIR="$OUT_BASE/actnet" python ACTNet-main/train.py \
      ACTNet-main/configs/actnet/actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_100e_bcd.py \
      --work-dir "$OUT_BASE/actnet_$DS" || echo "!!! ACTNet 训练失败"
    python ACTNet-main/test_cd.py \
      --data-base "$DATA_BASE" --dataset "$DS" \
      --config ACTNet-main/configs/actnet/actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_100e_bcd.py \
      --checkpoint "$OUT_BASE/actnet_$DS" --out-dir "$OUT_BASE/actnet" || echo "!!! ACTNet 评估失败"

    echo ">>> [$DS] FAEWNet"
    BCD_DATA_ROOT="$DATA_BASE/$DS" BCD_DATASET="$DS" \
      BCD_OUT_DIR="$OUT_BASE/faewnet" python FAEWNet-main/tools/train.py \
      FAEWNet-main/configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py \
      --work-dir "$OUT_BASE/faewnet_$DS" || echo "!!! FAEWNet 训练失败"
    python FAEWNet-main/test_cd.py \
      --data-base "$DATA_BASE" --dataset "$DS" \
      --config FAEWNet-main/configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py \
      --checkpoint "$OUT_BASE/faewnet_$DS" --out-dir "$OUT_BASE/faewnet" || echo "!!! FAEWNet 评估失败"
  fi
done

echo
echo "============================================================"
echo "全部完成。结果在 $OUT_BASE/<模型>/ 下，文件名带数据集名。"
echo "============================================================"
