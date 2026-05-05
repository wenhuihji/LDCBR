import numpy as np
import os

# 修改为你自己的数据路径
dataset_path = r"C:\Users\Administrator\Desktop\LDL-FLC-main\LDL-FLC-main\SJAFFE"
label_path = os.path.join(dataset_path, 'label.npy')

# 加载标签数据
Y = np.load(label_path)

# 检查标签维度
print("标签矩阵 shape:", Y.shape)
print("样本数:", Y.shape[0])
print("每个样本的标签个数（维度）:", Y.shape[1])
