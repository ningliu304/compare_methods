import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Type
import torch
import torch.nn as nn
from mmengine.runner import Runner
from mmengine import Config
from opencd.apis.opencd_inferencer_feature import OpenCDInferencerFeature

class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass

config_path = '/home/debian/Workspace/change_adapter/configs/actnet_smisup/actnet_intern_l_eva02_l_mit-b2_256x256_80k_whu_cd_20.py'
checkpoint_path = '/home/debian/Workspace/change_adapter/smi_sup/whu_cd_smi_sup_20/best_mIoU_iter_9000.pth'
dir_pre = '/public/datasets/WHU-CD-256/A/'
dir_post = '/public/datasets/WHU-CD-256/B/'

out_dir = '/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/'

cfg = Config.fromfile(config_path)
cfg.visualizer.save_dir = out_dir
inferencer = OpenCDInferencerFeature(model=config_path, weights=checkpoint_path,
                              classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]],)

dataset = Runner.build_dataloader(cfg.val_dataloader)
image_list = []
# 改图像类型，levir-cd为png，改为tif
for i in [file for file in os.listdir(dir_pre) if file.endswith('.png')]:
    image_list.append([os.path.join(dir_pre, i), os.path.join(dir_post, i)])

model = inferencer.model
# show all named module in the model and use it in source list below
for name, module in model.named_modules():
    print(name)

model.eval()

source = [
        # 'tf_encoder.layers.23',
        # 'decode_head.side_adapter_network.conv_clips.2.1',
        # 'decode_head.mask_decoder.conv_seg'
    ]
source = dict.fromkeys(source)

# for image in image_list:
#     recorder = Recorder()
#
#     for name, module in model.named_modules():
#         if name in source:
#             module.register_forward_hook(recorder.record_data_hook)
#     # 该路径只保存prediction，特征图在对应配置文件中：visualizer = dict(save_dir='/home/debian/open-cd-main/temp/')
#     with torch.no_grad():
#         inferencer(inputs=[image], img_out_dir=out_dir, recoder=recorder)
#
#     del image
#     del recorder
recorder = Recorder()
for name, module in model.named_modules():
    if name in source:
        module.register_forward_hook(recorder.record_data_hook)
inferencer(inputs=[['/public/datasets/WHU-CD-256/A/whucd_06791.png',
                          '/public/datasets/WHU-CD-256/B/whucd_06791.png']], img_out_dir=out_dir, recoder=recorder)