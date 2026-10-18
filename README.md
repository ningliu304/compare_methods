# 变化检测对比实验说明文档

本目录把 **11 个**开源变化检测（Change Detection, CD）模型统一到**同一数据、同一尺寸、
同一训练轮数、同一随机种子、同一指标实现**下，用于公平对比与复现。

模型：BIT_CD、ChangeFormer、TinyCD、Siam-NestedUNet(SNUNet)、DSAMNet、BCE-Net、CMCDNet、
ChangeViT、ChangeMamba、ACTNet、FAEWNet。（均为监督式方法）

---

## 0. 方法一览

按**参数量升序**排列。**时间**为论文发表的会议/期刊；**架构**为「骨干网络 + 核心机制」。

| 方法 | 参数量 (M) | 时间（发表） | 架构（骨干 + 核心机制） | 监督方式 |
|---|---|---|---|---|
| **TinyCD** | 0.29 | 2023, Neural Computing & Applications | EfficientNet-B4 轻量骨干 + 分组卷积特征混合 + 掩膜注意力解码 | 监督（二值） |
| **BIT** | 12.40 ᵃ | 2021, IEEE TGRS | ResNet 编码器 + Transformer 编/解码器（双时相语义 token）+ 特征差分 | 监督（二值） |
| **SNUNet-CD** | 12.03 ᵇ | 2021, IEEE GRSL | 密集连接 Siamese NestedUNet + ECAM 集成通道注意力 + 混合损失 | 监督（二值） |
| **DSAMNet** | 16.95 | 2021, IEEE TGRS | ResNet18 + CBAM 注意力 + 深度监督的度量（距离）模块 | 监督（二值） |
| **BCE-Net** | 28.77 ᶜ | 2023, ISPRS J. P&RS | ResNet34 + DCNv2 变形卷积对齐 + 多任务（新/旧/迁移）分支 + 对比学习 | 监督（单时相历史底图 + 新影像） |
| **ChangeFormer** | 41.03 ᵈ | 2022, IGARSS | 层次化 Transformer（Siamese）编码器 + MLP 解码器 | 监督（二值） |
| **ChangeMamba** | 32.61 ᵉ | 2024, IEEE TGRS | VMamba(SSM/状态空间) 编码器 + 时空关系建模解码器（BCD/SCD/BDA 三框架） | 监督（二值） |
| **ChangeViT** | 41.15 ᶠ | 2025, Pattern Recognition | 纯 ViT(DINOv2) 骨干 + detail-capture 细节捕获 + feature injector 特征注入 + 解码器 | 监督（二值） |
| **CMCDNet** | 待统计 ᵍ | 2023, Int. J. Appl. Earth Obs. Geoinf. | 双流 CNN（ResNet50 光学 + EfficientNet-B2 SAR）+ 门控融合 + 跨模态自注意力 | 监督（跨模态） |
| **FAEWNet** | 待统计 ᵍ | 2025, IEEE TGRS | SAM(ViT-L) 骨干 + LoRA(PEFT) + 分布感知傅里叶适配 + 边缘约束变形(SMSF) 时空融合 | 监督（二值） |
| **ACTNet** | 待统计 ᵍ | 2026, Pattern Recognition | CNN-Transformer 基础模型适配：InternImage(DCNv3) + EVA02 ViT + MiT-B2 侧网络 + SFFM 空频融合 | 监督（二值） |

> 参数量为**本实验实际配置下逐层统计**（含骨干），并已与论文/文献核对：
> ᵃ BIT 常被引作 3.55M，但该值不可能含 ResNet18 骨干（其自身≈11.2M），本文按实测 **12.40M** 计；
> ᵇ SNUNet-CD 官方主表为 **16 通道=3.0M**，本仓库硬编码 `n1=32`→**12.03M**（两者皆为已发表配置）；
> ᶜ BCE-Net 为改造版双时相 `model_bcd.py`（原方法单时相，论文未给参数量）；
> ᵈ ChangeFormer 用 **embed_dim=256**（=原文，41.0M）；若用仓库默认 64 则约 29.8M；
> ᵉ ChangeMamba 为 VSSM-**tiny** 配置；ᶠ ChangeViT 为 **small**(DINOv2 ViT-S) 配置（文献另有 32.1M 的记法，口径/配置不同）；
> ᵍ CMCDNet / FAEWNet / ACTNet 基于 mmseg 1.x（CMCDNet 为 0.20），需各自独立环境才能构建，
> 待环境就绪后用其 `tools/analysis_tools/get_flops.py <config>` 统计。
>
> **与原文一致**：DSAMNet 16.95M、TinyCD 0.28M、ChangeFormer 41.0M。

**来源**：
- BIT —《Remote Sensing Image Change Detection with Transformers》, TGRS 2021
- SNUNet-CD —《SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images》, GRSL 2021
- DSAMNet —《A Deeply Supervised Attention Metric-Based Network ...》, TGRS 2021
- ChangeFormer —《A Transformer-Based Siamese Network for Change Detection》, IGARSS 2022
- TinyCD —《TinyCD: A (Not So) Deep Learning Model for Change Detection》, NCAA 2023
- BCE-Net —《BCE-Net: Reliable Building Footprints Change Extraction ...》, ISPRS J. P&RS 2023
- CMCDNet —《Cross-modal change detection flood extraction based on CNN》, IJAEOG 2023
- ChangeMamba —《ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model》, TGRS 2024（arXiv:2404.03425）
- ChangeViT —《ChangeViT: Unleashing Plain Vision Transformers for Change Detection》, Pattern Recognition 2025（DOI:10.1016/j.patcog.2025.112539）
- FAEWNet —《SAM-Based Building Change Detection With Distribution-Aware Fourier Adaptation and Edge-Constrained Warping》, TGRS 2025（DOI:10.1109/TGRS.2025.3629110）
- ACTNet —《ACTNet: Adapting CNN-Transformer-based foundation model for RS change detection》, Pattern Recognition 2026

> 说明：为适配本实验的双时相二值变化设定，`BCE-Net` 使用改造版 `model_bcd.py`（原方法为单时相
> 「历史底图 + 新影像」语义变化）；`CMCDNet` 原为光学-SAR 跨模态，这里把 A/B 分别喂给 opt/sar 两分支。

---

## 1. 目录结构

```
E:\论文\对比实验\
├── common/                 # 共享模块（所有模型复用）
│   ├── config.py           # 统一命令行参数 / 配置（--dataset 等）
│   ├── datasets.py         # 数据集发现与切换
│   ├── seed.py             # 随机种子
│   ├── dataset.py          # 统一数据集
│   ├── metrics.py          # 统一指标（P/R/F1/IoU）
│   ├── checkpoint.py       # 权重枚举 / 跨格式加载
│   ├── adapter.py          # 模型适配器基类
│   ├── engine.py           # 统一训练/验证/测试引擎
│   ├── vis.py              # 推理可视化
│   └── logger.py           # 日志
├── run_all.sh              # 一键跑「所有模型 × 所有数据集」（新增）
├── data/                   # 数据集根目录：每个子目录一个数据集（见 §2）
│   ├── Evlab/{train,val,test}/{A,B,label}
│   ├── HRSCD/{train,val,test}/{A,B,label}
│   └── WUSU512/{train,val,test}/{A,B,label}
├── BIT_CD-master/
│   ├── adapter.py          # 模型适配器（新增）
│   ├── train_cd.py         # 统一入口（新增）
│   └── ...                 # 原仓库文件保持不变
├── ChangeFormer-main/{adapter.py, train_cd.py, ...}
├── Tiny_model_4_CD-main/{adapter.py, train_cd.py, ...}
├── Siam-NestedUNet-master/{adapter.py, train_cd.py, ...}
├── DSAMNet-main/{adapter.py, train_cd.py, ...}
├── BCE-Net-main/{model_bcd.py, adapter.py, train_cd.py, ...}
├── CMCDNet-main/
│   ├── my_scripts/bcd_rgb/ # MMSeg 配置（新增）
│   ├── test_bcd.py         # 统一评估入口（新增）
│   └── ...
├── ACTNet-main/
│   ├── configs/actnet/actnet_..._100e_bcd.py  # 100e+seed 配置（新增）
│   ├── test_cd.py          # 统一评估入口（新增）
│   └── ...
├── FAEWNet-main/
│   ├── configs/faewnet/faewnet_..._100e_bcd.py  # 100e+seed 配置（新增）
│   ├── test_cd.py          # 统一评估入口（新增）
│   └── ...
├── ChangeViT-master/{adapter.py, train_cd.py, ...}   # 新增
└── ChangeMamba-master/{adapter.py, train_cd.py, ...} # 新增
```

> 设计原则：**原仓库代码不改**，只新增“适配器 + 入口脚本”，共享逻辑集中在 `common/`。
> 想改整体流程只改 `common/`，想改某个模型只改它目录下的 `adapter.py`。

---

## 2. 数据格式

每个数据集的结构固定为：

```
data/
├── Evlab/{train,val,test}/{A,B,label}
├── HRSCD/{train,val,test}/{A,B,label}
└── WUSU512/{train,val,test}/{A,B,label}
```

即 `data/<数据集名>/` 下含 `train / val / test`，每个 split 含 `A / B / label`。

规则：
- A、B 各为 **3 通道 RGB**，`label` 为单通道二值图（`0`=背景，`1`/`255`=变化）。
- 同一 split 内按**文件名主干**配对：如 `train/A/001.png`、`train/B/001.png`、`train/label/001.png`，
  扩展名可不同。
- 图片会统一缩放到 **512×512**（图像双线性、标签最近邻）。
- 标签目录名兼容 `label / labels / L / OUT / gt / mask` 等常见写法。

### 2.1 切换数据集（重点）

新增数据集 = **往 `data/` 下放一个新文件夹**（结构同上），无需改任何代码。
运行时用 `--dataset <名称>` 选中即可：

```bash
# 查看当前有哪些数据集
python BIT_CD-master/train_cd.py --list-datasets

# 切到某个数据集
python BIT_CD-master/train_cd.py --dataset Evlab   --out-dir runs/bit_cd
python BIT_CD-master/train_cd.py --dataset WUSU512 --out-dir runs/bit_cd
```

- 数据根目录默认是 `data/`；可用 `--data-base` 改成别的目录。
- 也可用 `--root <目录>` 直接指定某个数据集的根目录（优先级最高）。
- MMSeg 系（CMCDNet/ACTNet）用环境变量切换：`BCD_DATA_ROOT=data/Evlab python tools/train.py ...`。
- 一键跑「所有模型 × 所有数据集」：`bash run_all.sh`（详见 §5.4）。

> **命名约定**：所有产物都会带上数据集名，互不覆盖。例如数据集 `Evlab` 的训练产物为
> `runs/bit_cd/best_Evlab.pth`、`runs/bit_cd/log_Evlab.txt`、`runs/bit_cd/vis_Evlab/`。

---

## 3. 统一约定

| 项目 | 统一设置 | 说明 |
|---|---|---|
| 输入尺寸 | 512×512 | `--img-size` 可改 |
| 训练轮数 | 100 | `--epochs` 可改 |
| 随机种子 | 42 | `--seed` 可改，固定 random/numpy/torch/cuda + cuDNN |
| 数据增强 | 随机水平/垂直翻转 + 90° 旋转 | 仅训练集 |
| 指标 | Precision / Recall / F1 / IoU（变化类） | 全部模型逐字复用官方 `ConfuseMatrixMeter` |
| 最优模型选择 | **验证集变化类 F1 最高**（不用均值 mF1） | 存为 `best.pth`（MMSeg 系见 §5.3/§5.2） |
| 测试 | 用 `best.pth` 在测试集评估 + 可视化 | |
| 归一化 | **默认沿用各模型原实现** | `--norm` 可强制统一（见 §3.1） |
| 二值化 | **各模型保留原设计** | 见 §3.2 |

### 3.1 归一化（`--norm`）
为避免人为削弱某个模型，默认各自使用**原文的预处理**：

| 模型 | 默认 `norm` | 等价处理 |
|---|---|---|
| BIT_CD / ChangeFormer / DSAMNet | `minus_one_one` | `[-1, 1]` |
| Siam-NestedUNet / BCE-Net / CMCDNet | `zero_one` | `[0, 1]` |
| TinyCD | `imagenet` | ImageNet mean/std |
| ChangeViT | `imagenet` | ImageNet mean/std（BGR 顺序，匹配其 cv2 读图） |
| ChangeMamba | `imagenet` | ImageNet mean/std（0~255 尺度，与其 imutils 等价） |
| ACTNet / FAEWNet | （由模型 `data_preprocessor` 内部处理） | 0~255，mean/std=文中值，bgr_to_rgb |

若要做**严格公平消融**，所有模型都加 `--norm imagenet` 即可全局统一。
（基于 MMSeg 的 ACTNet/CMCDNet 不受该开关影响，各自保留原文预处理。）

### 3.2 二值化规则（论文中务必注明）
不同模型的输出语义不同，二值化在各模型的 `adapter.py` 内实现：

| 模型 | 输出 | 二值化 |
|---|---|---|
| BIT_CD / ChangeFormer / Siam-NestedUNet | `[B,2,H,W]` logits | `argmax` |
| TinyCD | `[B,1,H,W]` sigmoid 概率 | `> 0.5` |
| BCE-Net | `[B,1,H,W]` logits | `sigmoid > 0.5` |
| DSAMNet | 距离图 `dist` | `dist > 1.0`（原设计） |
| ChangeViT | `[B,1,H,W]` sigmoid 概率 | `> 0.5` |
| CMCDNet / ACTNet / FAEWNet / ChangeMamba | `[B,2,H,W]` logits | `argmax` |

---

## 4. 环境依赖

- **通用**：Python 3.8+、PyTorch、torchvision、numpy、Pillow、tqdm。
- **BIT_CD / ChangeFormer**：额外需 `einops`。
- **TinyCD**：首次运行会下载 EfficientNet-B4 预训练权重。
- **BCE-Net**：
  - 需要 `torchvision` 的 resnet34 预训练权重（首次自动下载）。
  - **DCNv2 为可选**：若已编译（`cd BCE-Net-main/DCNv2 && python setup.py build_ext --inplace`）
    则使用原可变形卷积；未编译会自动退化为普通卷积，仍可训练。
- **CMCDNet**：
  - 依赖 **mmcv-full 1.3~1.4**（自定义编译）+ `timm` + `prettytable`，否则 `test_bcd.py` 会明确报错。
- **ACTNet**：
  - 基于 **OpenCD / mmsegmentation 1.x**，依赖 `mmcv>=2.0.0rc4,<2.1.0`、`mmseg>=1.0.0rc6`、
    `mmdet>=3.0.0rc6`、`mmengine`、`timm`。
  - 需编译自定义算子 **DCNv3**（`cd ops_dcnv3 && sh make.sh`）与 **DCNv4**（`DCNv4_op/make.sh`）。
  - 预训练权重（InternImage-XL / EVA02-L / MiT-B2）需按本地路径修改配置。
- **FAEWNet**：
  - 基于 **OpenCD / mmsegmentation 1.x**，依赖 `mmcv>=2.0.0`、`mmseg>=1.0.0rc6`、`mmpretrain>=1.0.0rc7`、
    `mmdet>=3.0.0`、`mmengine`、`peft`。
  - SAM ViT-L 预训练权重（`vit-large-p16_sam-...`）由 `init_cfg` 自动下载。
- **ChangeViT**：
  - 依赖 `einops`（必需）、`torchvision`；`xformers` 可选（缺失时自动回退普通注意力）。
  - 需把预训练权重放到 `ChangeViT-master/checkpoint/`（`small`: `dinov2_vits14_pretrain.pth`；
    `tiny`: `deit_tiny_patch16_224-a1311bcf.pth`）。
- **ChangeMamba**：
  - 需编译 SSM 自定义算子：`cd ChangeMamba-master/kernels/selective_scan && pip install .`
    （需 CUDA≥11.6、nvcc、ninja、einops），否则 import 会报 `selective_scan_cuda` 缺失。
  - 可选：设环境变量 `CHANGEMAMBA_PRETRAINED=/path/to/vssm_tiny_ckpt.pth` 加载 VMamba 预训练编码器。

> **版本冲突提示（重要）**：`CMCDNet`、`ACTNet`、`FAEWNet`、`ChangeMamba` 对底层库版本要求互不兼容，
> 建议**各自建独立 conda 环境**，不要与其余 7 个模型共用同一个 env：
> - **CMCDNet** 需 `mmcv-full 1.3~1.5`（mmseg 0.20 fork）；
> - **ACTNet** 需 `mmcv 2.0~2.1` + `mmseg 1.x` + 编译 DCNv3/DCNv4 算子；
> - **FAEWNet** 需 `mmcv 2.0~2.1` + `mmseg 1.x` + `mmpretrain`/`mmdet`/`peft`；
> - **ChangeMamba** 依赖 `triton`（**无官方 Windows 版，需 Linux/WSL**）+ 编译 `kernels/selective_scan`。
>
> 其余 7 个模型（BIT / ChangeFormer / TinyCD / SNUNet / DSAMNet / BCE-Net / ChangeViT）
> 可在同一环境直接运行（已验证）。

---

## 5. 运行方法

> 以下命令均从 `E:\论文\对比实验` 根目录执行。首次跑通建议加 `--epochs 1` 做冒烟测试。

### 5.1 标准模型（8 个，统一 Trainer）

```bash
python BIT_CD-master/train_cd.py        --dataset Evlab --out-dir runs/bit_cd
python ChangeFormer-main/train_cd.py    --dataset Evlab --out-dir runs/changeformer
python Tiny_model_4_CD-main/train_cd.py --dataset Evlab --out-dir runs/tinycd
python Siam-NestedUNet-master/train_cd.py --dataset Evlab --out-dir runs/snunet
python DSAMNet-main/train_cd.py         --dataset Evlab --out-dir runs/dsamnet
python BCE-Net-main/train_cd.py         --dataset Evlab --out-dir runs/bcenet
python ChangeViT-master/train_cd.py     --dataset Evlab --out-dir runs/changevit
python ChangeMamba-master/train_cd.py   --dataset Evlab --out-dir runs/changemamba
```
（把 `--dataset Evlab` 换成 `HRSCD` / `WUSU512` 即切换数据集。）

> ChangeViT 需先在 `ChangeViT-master/checkpoint/` 放好 DINOv2 权重；ChangeMamba 需先编译
> `kernels/selective_scan`（详见 §4）。两者其余用法与上面完全一致，最优模型按验证集变化类 F1 保存。

常用参数（所有模型通用）：

```bash
--root data          # 数据根目录
--out-dir runs/xxx   # 输出目录（也可用各自的默认值）
--epochs 100         # 训练轮数
--batch-size 8       # 批大小（显存不足就调小）
--lr 1e-4            # 学习率
--seed 42            # 随机种子
--gpu 0              # GPU 编号
--norm imagenet      # 强制统一归一化（可选）
--num-vis 8          # 测试集可视化样本数
```

冒烟测试示例：
```bash
python BIT_CD-master/train_cd.py --dataset Evlab --out-dir runs/_smoke --epochs 1 --batch-size 2
```

### 5.2 CMCDNet（MMSeg 流程）

训练走 MMSeg 原生入口（100 epoch、seed 42 已写入配置）。用环境变量 `BCD_DATA_ROOT` 切换数据集：
```bash
BCD_DATA_ROOT=data/Evlab python CMCDNet-main/tools/train.py \
  CMCDNet-main/my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py \
  --seed 42 --deterministic --work-dir runs/cmcd_Evlab
```

训练完成后，用统一口径评估并可视化（复用 `common` 的指标与可视化，与其他模型完全一致）：
```bash
# 传目录：自动在验证集上按「变化类 F1」挑选最优权重，再在测试集报告
python CMCDNet-main/test_bcd.py --dataset Evlab \
  --config CMCDNet-main/my_scripts/bcd_rgb/bcd_rgb_cmcd_r50-r18.py \
  --checkpoint runs/cmcd_Evlab \
  --out-dir runs/cmcd
```
> 评估产物同样带数据集名：`runs/cmcd/best_Evlab.pth`、`runs/cmcd/vis_Evlab/`。
> 训练已改为每 5 epoch 存一次权重（不再用 `save_best='mIoU'`）；最优权重由 `test_bcd.py`
> 按**验证集变化类 F1** 挑选，与其余模型口径一致。也可直接传单个 `.pth` 跳过挑选。

### 5.3 ACTNet（OpenCD / MMSeg 1.x 流程）

训练走 OpenCD 原生入口（100 epoch、seed=42 已写入配置）。用环境变量 `BCD_DATA_ROOT` 切换数据集：
```bash
BCD_DATA_ROOT=data/Evlab python ACTNet-main/train.py \
  ACTNet-main/configs/actnet/actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_100e_bcd.py \
  --work-dir runs/actnet_Evlab
```

训练完成后用统一口径评估并可视化（传目录时按验证集变化类 F1 自动挑选最优权重）：
```bash
python ACTNet-main/test_cd.py --dataset Evlab \
  --config ACTNet-main/configs/actnet/actnet_intern_l_eva02_l_mit-b2_msk_head_512x512_100e_bcd.py \
  --checkpoint runs/actnet_Evlab --out-dir runs/actnet
```
> 训练已改为每 5 epoch 存一次权重（不再用 `save_best='mIoU'`）；最优权重由 `test_cd.py`
> 按**验证集变化类 F1** 挑选。若 `test_cd.py` 因环境 API 差异报错，可改用原生
> `python test.py <config> <ckpt>`（其 mFscore/mIoU 与 `common` 的 P/R/F1/IoU 公式一致）。

### 5.4 FAEWNet（OpenCD / MMSeg 1.x 流程）

训练走 OpenCD 原生入口（100 epoch、seed=42 已写入配置）。用环境变量 `BCD_DATA_ROOT` 切换数据集：
```bash
BCD_DATA_ROOT=data/Evlab python FAEWNet-main/tools/train.py \
  FAEWNet-main/configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py \
  --work-dir runs/faewnet_Evlab
```

训练完成后用统一口径评估并可视化（传目录时按验证集变化类 F1 自动挑选最优权重）：
```bash
python FAEWNet-main/test_cd.py --dataset Evlab \
  --config FAEWNet-main/configs/faewnet/faewnet_vit-sam-l_512x512_fp16-100e_bcd.py \
  --checkpoint runs/faewnet_Evlab --out-dir runs/faewnet
```

### 5.5 一键跑「所有模型 × 所有数据集」

```bash
bash run_all.sh                                  # 8 个 Trainer 模型 × 全部数据集
DATASETS="Evlab" bash run_all.sh                 # 只跑一个数据集
EPOCHS=1 bash run_all.sh                         # 冒烟测试
WITH_MMSEG=1 bash run_all.sh                     # 额外跑 CMCDNet / ACTNet / FAEWNet
UNIFIED="--norm imagenet --lr 1e-4 --optim adamw" bash run_all.sh   # 强制统一超参
```

### 5.6 冒烟测试（确认能跑通，不跑满）

用 `--epochs 1` + `--max-batches N` 每个 split 只跑几个 batch，几秒~几分钟即可验证整条链路：

```bash
python BIT_CD-master/train_cd.py --dataset Evlab --out-dir runs/_smoke --epochs 1 --batch-size 2 --max-batches 2
python ChangeViT-master/train_cd.py --dataset Evlab --out-dir runs/_smoke --epochs 1 --batch-size 1 --max-batches 2
```
> 注意：512 下 `ChangeViT` 等大模型在 8GB 显存需 `--batch-size 1`，否则会 OOM。

---

## 6. 输出文件

每个模型的输出目录（如 `runs/bit_cd/`）下，**文件名都带数据集名**，因此多数据集互不覆盖：

```
runs/bit_cd/
├── log_Evlab.txt        # 训练日志：每轮一行 epoch | train_loss | val_loss | Precision | Recall | F1 | IoU
├── best_Evlab.pth       # 验证集「变化类 F1」最优的模型权重
├── vis_Evlab/           # 测试集可视化：A｜B｜GT｜Pred｜TP/FP/FN 五联图
│   └── 0000_xxx.png
├── log_HRSCD.txt        # 换数据集后自动另存，不覆盖上面的
├── best_HRSCD.pth
└── vis_HRSCD/
```

测试结束时控制台会打印最终指标（含数据集名）：

```
============================================================
[TEST] BIT_CD @ Evlab
  Precision=0.xxxx  Recall=0.xxxx  F1=0.xxxx  IoU=0.xxxx
  mIoU=0.xxxx  mF1=0.xxxx  OA=0.xxxx
  visualizations -> runs/bit_cd/vis_Evlab
============================================================
```

---

## 7. 如何扩展 / 改进

### 新增一个模型
1. 在模型目录下写 `adapter.py`，继承 `common.adapter.CDAdapter`，实现 5 个钩子：
   - `build_model(cfg)`：构建网络
   - `prepare_batch(batch, device)`：把 batch 转成 `(inputs, target)`（指定 target 的 dtype）
   - `forward(model, inputs)`：返回含 `'prob'` 的 dict
   - `compute_loss(model, out, target)`：返回标量 loss
   - `predict_binary(out, inputs)`：返回 `[B,H,W]` 的 0/1 图
2. 照抄 `train_cd.py` 模板（约 30 行），改 `defaults` 里的超参即可。

### 改进整体流程
只改 `common/` 即可对所有模型生效，例如：
- 加新指标 → `common/metrics.py`
- 换优化器/调度器 → `common/engine.py`
- 换增强策略 → `common/dataset.py`

### 代码分工
| 层 | 位置 | 职责 |
|---|---|---|
| 网络定义 | 各模型目录（原仓库） | 保持原样，不改 |
| 适配层 | 各模型目录 `adapter.py` | forward / loss / 二值化 |
| 通用层 | `common/` | 数据、指标、训练循环、可视化、日志 |
| 入口层 | 各模型目录 `train_cd.py` | 拼装并启动 |

> 例外：MMSeg 系（CMCDNet、ACTNet、FAEWNet）训练走各自原生入口（配置文件在各自目录），
> 统一评估用 `test_bcd.py` / `test_cd.py`，在评估阶段复用 `common` 的指标与可视化，保证口径一致。

---

## 8. 常见问题

- **显存不足（OOM）**：调小 `--batch-size`（BCE-Net 默认 4，其余默认 8/16）。
- **怎么换数据集**：往 `data/` 下放新文件夹（结构同 §2），运行时加 `--dataset <名称>` 即可；
  `--list-datasets` 可列出已有数据集。MMSeg 系用 `BCD_DATA_ROOT=data/<名称>` 切换。
- **找不到数据**：确认 `--dataset <名称>` 对应的 `data/<名称>/train/A` 存在（子目录名固定为 `A/B/label`）；
  也可用 `--root <目录>` 直接指定，或用 `--list-datasets` 排查。
- **BCE-Net 提示 DCNv2**：属正常，未编译 DCNv2 时会自动用普通卷积兜底，不影响运行。
- **CMCDNet 报 mmseg/mmcv 错误**：需先按仓库文档编译安装匹配的 mmcv-full 1.3~1.4。
- **ACTNet 报 DCNv3/DCNv4 错误**：需先编译 `ops_dcnv3` 与 `DCNv4_op`；并确认 mmcv 2.x、mmseg 1.x 版本匹配。
- **ACTNet 找不到预训练权重**：配置里 internimage_xl / eva02-large / mit_b2 的路径为原作者的绝对路径，需改成你的本地路径。
- **FAEWNet 报 mmseg/mmcv 错误**：需 `mmseg>=1.0.0rc6` + `mmcv 2.0~2.1` + `mmpretrain`/`mmdet`/`peft`；SAM ViT-L 权重会自动下载。
- **ChangeViT 找不到 ViT 权重**：需把 `dinov2_vits14_pretrain.pth`（small）或
  `deit_tiny_patch16_224-a1311bcf.pth`（tiny）放到 `ChangeViT-master/checkpoint/`。
- **ChangeMamba 报 selective_scan_cuda 缺失**：需先编译 `kernels/selective_scan`；不加载
  VMamba 预训练权重也能跑（随机初始化），但精度会低很多，建议设 `CHANGEMAMBA_PRETRAINED`。
- **最优模型怎么选**：统一按**验证集变化类 F1**（不是均值 mF1）。8 个 Trainer 模型在训练中自动存 `best.pth`；
  MMSeg 系（CMCDNet/ACTNet）每 5 epoch 存权重，由 `test_bcd.py`/`test_cd.py` 传目录时按 F1 挑选。
- **想要绝对可复现**：固定 `--seed`，并尽量在同一硬件/同一 cuDNN 环境下运行。
