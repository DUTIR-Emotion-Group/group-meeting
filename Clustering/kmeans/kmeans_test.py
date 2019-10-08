# -*- encoding:utf-8 -*-

from kmeans import *
import matplotlib.pyplot as plt

dataMat = file2matrix('4k2_far.txt', "\t") # 从文件构建的数据集
dataSet = dataMat[:, 1:] # 提取数据集中的特征列

k = 4 # 外部指定1,2,3...通过观察数据集有4个聚类中心
clustercents, ClustDist = kMeans(dataSet, k)

# 返回计算完成的聚类中心
print("clustercents:\n", clustercents)

# 输出生成的ClustDist：对应的聚类中心(列1),到聚类中心的距离(列2),行与dataSet一一对应
color_cluster(ClustDist[:, 0:1], dataSet, plt)

# 绘制聚类中心
drawScatter(plt, clustercents, size=60, color='red', mrkr='D')
plt.show()