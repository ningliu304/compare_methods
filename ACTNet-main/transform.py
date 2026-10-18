# from PIL import Image
# import numpy as np
# import os
#
# # 1.文件夹A存放真实标签，B存放预测标签，C存放处理后的图片
# # 2.A、B文件夹中对应图像形状默认一致
# # 3.误标的像素设为红色，漏标的像素设为绿色
#
# # 文件夹路径
# folder_A = "/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/gt/"
# folder_B = "/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/pred/"
# folder_C = r"/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/processed/"
#
# # 确保文件夹C存在
# if not os.path.exists(folder_C):
#     os.makedirs(folder_C)
#
# # 获取文件夹A中的所有图像文件名
# img_files_A = os.listdir(folder_A)
#
# # 对比A和B文件夹中的图像
# for img_file in img_files_A:
#     img_path_A = os.path.join(folder_A, img_file)
#     img_path_B = os.path.join(folder_B, img_file)
#     img_path_C = os.path.join(folder_C, img_file)
#
#     # 打开图像
#     img_A = Image.open(img_path_A).convert('L')
#     img_B = Image.open(img_path_B).convert('L')
#
#     img_A = img_A.convert('RGB')  # 三通道，每个通道值都相同
#     img_B = img_B.convert('RGB')  # 三通道，每个通道值都相同
#
#     # 处理图像
#     for x in range(img_A.width):
#         for y in range(img_A.height):
#             pixel_A = img_A.getpixel((x, y))
#             pixel_B = img_B.getpixel((x, y))
#
#             # 误检的像素设为红色，漏检的像素设为绿色
#             if pixel_A == (0, 0, 0) and pixel_B == (255, 255, 255):
#                 img_B.putpixel((x, y), (255, 0, 0))
#             elif pixel_A == (255, 255, 255) and pixel_B == (0, 0, 0):
#                 img_B.putpixel((x, y), (0, 255, 0))
#
#     # 保存处理后的图像到文件夹C
#     img_B.save(img_path_C)
#
# print("处理完成！")


# from PIL import Image
# import numpy as np
#
# # 1. 加载您的图1 (假设文件名为 image1.png)
# img = Image.open('/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/whucd_07413.png').convert('RGB')
# data = np.array(img)
#
# # 2. 定义蓝色的识别范围（针对图中的浅蓝色）
# # 我们允许一定的色差范围以确保完全覆盖
# r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
# blue_mask = (r > 100) & (r < 150) & (g > 180) & (g < 220) & (b > 200)
#
# # 3. 将蓝色部分替换为图2中的纯绿色 [0, 255, 0]
# data[blue_mask] = [0, 255, 0]
#
# # 4. 保存并查看结果
# result = Image.fromarray(data)
# result.save('/home/debian/Workspace/change_adapter/smi_sup/whu_cd_20/transformed_image.png')
# result.show()


import numpy as np
from scipy.stats import ttest_ind

# 输入: 2×3矩阵
# 第1行: 方法A在3次重复实验中的结果
# 第2行: 方法B在3次重复实验中的结果

# data = np.array([
#     [92.47, 92.35, 92.53],  # 类别/方法1
#     [92.28, 92.28, 92.16]   # 类别/方法2
# ]) # Levircd MLP vs Intern

# data = np.array([
#     [92.24, 92.28, 92.23],  # 类别/方法1
#     [92.28, 92.28, 92.16]   # 类别/方法2
# ]) # Levricd Mask MLP vs Intern

# data = np.array([
#     [92.24, 92.28, 92.23],  # 类别/方法1
#     [92.18, 92.28, 92.15]   # 类别/方法2
# ]) # Levricd Mask vs eva

# data = np.array([
#     [92.47, 92.35, 92.53],  # 类别/方法1
#     [92.18, 92.28, 92.15]   # 类别/方法2
# ]) # Levricd MLP vs eva

# data = np.array([
#     [95.05, 94.94, 94.86],  # 类别/方法1
#     [94.17, 94.59, 93.44]   # 类别/方法2
# ]) # WHU-CD MLP vs Intern

# data = np.array([
#     [95.55, 95.46, 95.41],  # 类别/方法1
#     [94.17, 94.59, 93.44]   # 类别/方法2
# ]) # WHU-CD Mask vs Intern

# data = np.array([
#     [95.05, 94.94, 94.86],  # 类别/方法1
#     [94.11, 94.29, 93.67]   # 类别/方法2
# ]) # WHU-CD MLP vs EVA

# data = np.array([
#     [95.55, 95.46, 95.41],  # 类别/方法1
#     [94.11, 94.29, 93.67]   # 类别/方法2
# ]) # WHU-CD Mask vs EVA

# data = np.array([
#     [82.34, 82.50, 82.34],  # 类别/方法1
#     [81.22, 81.57, 81.13]   # 类别/方法2
# ]) # SYSU MLP vs Intern

# data = np.array([
#     [82.34, 82.50, 82.34],  # 类别/方法1
#     [79.50, 80.89, 80.23]   # 类别/方法2
# ]) # SYSU MLP vs EVA

# data = np.array([
#     [84.88, 84.55, 84.84],  # 类别/方法1
#     [81.22, 81.57, 81.13]   # 类别/方法2
# ]) # SYSU Mask vs Intern

# data = np.array([
#     [84.88, 84.55, 84.84],  # 类别/方法1
#     [79.50, 80.89, 80.23]   # 类别/方法2
# ]) # SYSU Mask vs EVA

# data = np.array([
#     [67.75, 67.66, 67.80],  # 类别/方法1
#     [66.34, 65.43, 66.33]   # 类别/方法2
# ]) # s2looking MLP vs eva

# data = np.array([
#     [67.49, 67.55, 67.36],  # 类别/方法1
#     [66.34, 65.43, 66.33]   # 类别/方法2
# ]) # s2looking Mask vs eva

# data = np.array([
#     [67.49, 67.55, 67.36],  # 类别/方法1
#     [67.18, 66.84, 67.11]   # 类别/方法2
# ]) # s2looking Mask vs intern

data = np.array([
    [67.75, 67.66, 67.80],  # 类别/方法1
    [67.18, 66.84, 67.11]   # 类别/方法2
]) # s2looking MLP vs intern

group1 = data[0]
group2 = data[1]

# 计算均值和标准差
mean1 = np.mean(group1)
std1 = np.std(group1, ddof=1)   # 样本标准差

mean2 = np.mean(group2)
std2 = np.std(group2, ddof=1)

# t 检验
t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

print(f"Group 1: {group1}")
print(f"Mean ± Std: {mean1:.4f} ± {std1:.4f}")

print(f"Group 2: {group2}")
print(f"Mean ± Std: {mean2:.4f} ± {std2:.4f}")

print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Difference is statistically significant (p < 0.05).")
else:
    print("Difference is not statistically significant (p >= 0.05).")