# Liuzhaoxi 2023/11/7 16:16
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
data = np.genfromtxt('data_loader/PeMS-M/PeMS-M/W_228.csv', delimiter=',')

# 设置全局字体样式
plt.rcParams['font.family'] = 'Times New Roman'
# 创建矢量图
plt.figure(figsize=(6, 4))
plt.imshow(data, cmap='Blues', interpolation='nearest')
plt.colorbar(label='')
plt.title('Adjacency Matrix')
plt.xlabel('Nodes')
plt.ylabel('Nodes')

# 保存为SVG文件
plt.savefig('adjacency_matrix.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
