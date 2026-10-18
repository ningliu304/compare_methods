# CMCDNet 中文说明文档

本文档基于仓库内实现代码整理，对应论文：[Cross-modal change detection flood extraction based on convolutional neural network](https://www.sciencedirect.com/science/article/pii/S1569843223000195)（《International Journal of Applied Earth Observation and Geoinformation》，2023）。

## 项目概述

**CMCDNet** 面向**跨模态变化检测**：输入为**灾前光学影像**与**灾后 SAR（VV）影像**，输出为**洪水淹没区域**的二值语义分割结果。实现建立在 **MMSegmentation v0.20.0** 之上，通过自定义骨干网络 `CMCD` 与数据集类 `WCDataset` 完成光学–SAR 联合编码与解码。

## 代码结构（与本项目直接相关的部分）

| 路径 | 作用 |
|------|------|
| `mmseg/models/backbones/cmcd.py` | CMCD 骨干：`CMCD` 类及 RCB、CAB、注意力融合等子模块 |
| `mmseg/datasets/wc_dataset.py` | `WCDataset`：读取光学主图、SAR 辅助图与洪水标注 |
| `mmseg/datasets/pipelines/water_change/__init__.py` | 数据流水线：`wc_LoadImageFromFile`、`wc_StackByChannel`、`wc_Normalize` 等 |
| `my_scripts/water_change/` | 训练配置：`opt_sar_cmcd_r50-effb2_30e.py` 及 `base/` 下的数据集、调度与运行时 |

其余目录大体为 MMSegmentation 上游框架代码（`tools/`、`configs/`、`tests/` 等）。

## 模型原理简述（与 `cmcd.py` 对应）

`CMCD` 将批次输入按通道拆分为：

- **光学分支**：前 **4** 个通道（通常为 RGB + 第 4 波段，由配置 `backbone_opt_cfg.in_channels=4` 决定）
- **SAR 分支**：**1** 个通道（灾后 VV）

主要流程：

1. **双编码器**：光学侧默认 **ResNet50**（timm），SAR 侧在示例配置中为 **EfficientNet-B2**（timm）；各输出 5 级特征金字塔。
2. **桥接**：最深一层特征可选 **Dblock**（多尺度空洞卷积残差）增强感受野（`center_block="dblock"`）。
3. **侧向瓶颈（RCB）**：把各尺度编码特征统一到 `side_dim`（默认 64）通道。
4. **跨模态融合**：浅层四级使用 **GatedFusion**（门控加权融合光学与 SAR）；**最深层**使用 **AttFuseBlock**，内含 **SpatialAttBlock** 与 **ChannelAttBlock**（基于 QKV 的空间/通道注意力），再相加融合。
5. **双解码**：光学与 SAR 两侧各自 **DecoderBlock**（上采样 + 深度可分离卷积 DSC）与 **AlignDecoderBlock**（内含 **CAB**：可学习的双线性网格采样对齐高低层特征），自顶向下逐级融合。
6. **输出**：对两侧解码结果再做 **GatedFusion** 得到最终特征图 `(f_final,)`，供 MMSeg 的 `EncoderDecoder` + `FCNHead` 做像素分类。

可调要点（构造参数）：`enc_opt_dims` / `enc_sar_dims` 需与所选 timm 骨干各 stage 通道一致；`side_dim`、`att_dim_factor`、`center_block` 控制宽度与是否使用 Dblock。

## 数据与标注

示例配置 `my_scripts/water_change/base/opt_sar_inc_vv_dataset.py` 约定：

- **`data_root`**：默认 `../data`（相对于运行 `tools/train.py` 时的当前工作目录，一般为项目根下的 `data`）。
- **类别**：`("bg", "increase")`，共 **2** 类；`increase` 表示洪水新增水体等区域。
- **目录结构**（与英文 README 一致）：

```
data/
├── train/
│   ├── opt/       # 灾前光学影像
│   ├── vv/        # 灾后 SAR（VV），与 opt 文件名对齐
│   └── flood_vv/  # 与 SAR 对齐的洪水语义标注
└── test/
    ├── opt/
    ├── vv/
    └── flood_vv/
```

**说明**：若使用 `split` 文本列表抽样，需在 `WCDataset` 配置中指定 `split` 路径；当前 base 配置为扫描目录下全部影像。

## 数据流水线说明

- **`wc_LoadImageFromFile`**：同时读取 `img_dir` 与 `aux_dir` 下同名的光学图与 SAR 图，写入 `img`、`aux`。
- **`wc_StackByChannel`**：将光学与 SAR 在通道维拼接为 **5** 通道张量，作为 `CMCD.forward` 中 `torch.split(..., (4, 1))` 的输入。
- **`wc_Normalize`**：将图像转为 `float32` 并除以 **255**（简单缩放到 \[0,1\]）。
- **训练增强**：随机翻转、随机旋转（90°）；验证/测试使用 `MultiScaleFlipAug` 与固定 `img_scale`。

## 训练配置摘要（`opt_sar_cmcd_r50-effb2_30e.py`）

- **优化器**：Adam，学习率 **2e-4**（覆盖 base 中的 1e-3）。
- **Batch**：`samples_per_gpu=4`，`workers_per_gpu=4`（覆盖 dataset base 中的较大默认值）。
- **解码头**：`FCNHead`，**2** 类；损失为 **CrossEntropyLoss** + **DiceLoss**，类别权重均为 `(1.0, 2.0)`（略偏重洪水类）。
- **归一化**：骨干使用 **SyncBN**（单机单卡时需与环境兼容；若报错可查阅 MMSeg 文档改为 `BN`）。
- **调度**：30 epoch，Step LR（约在第 20、25 epoch 衰减），详见 `base/scheduler_30e.py`。

## 环境安装与运行

1. 按 [MMSegmentation v0.20.0 安装说明](https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/docs/get_started.md#installation) 安装 **mmcv**、**mmsegmentation** 及依赖（含 **timm**，用于 `TIMMBackbone`）。
2. 克隆本仓库，创建 `data` 目录，按英文 README 或上文结构放置 **CAU-Flood**（或自建同源结构）数据。
3. 在项目根目录执行（路径以官方脚本为准；若仓库根即含 `tools/`，则 `cd` 到该根目录）：

**单卡训练：**

```bash
python tools/train.py my_scripts/water_change/opt_sar_cmcd_r50-effb2_30e.py
```

**多卡训练（Linux 常见写法）：**

```bash
bash tools/dist_train.sh my_scripts/water_change/opt_sar_cmcd_r50-effb2_30e.py <GPU数量>
```

Windows 下若无 `bash`，可使用 MMSeg 文档中的 `torch.distributed.launch` 等价命令启动分布式训练。

可根据算力在配置中修改 `samples_per_gpu`、`img_scale` 以及 `backbone_*_cfg`（需同步修改 `enc_*_dims`）。

## 论文引用

若使用本仓库或模型思路，请引用原论文（BibTeX 见英文 `README.md`）。

## 与英文 README 的差异说明

- 英文 README 中数据集名称一处写作 “CHU-Flood”，下载链接描述为 “CAU-Flood”，请以实际下载页面与解压目录为准。
- 英文 README 中训练步骤写有 `cd cmcdnet`，当前树一般以仓库根为工程根目录；`data_root='../data'` 表示从包含 `tools` 的目录的上一级查找 `data`，请在正确的工程根下执行训练命令。
