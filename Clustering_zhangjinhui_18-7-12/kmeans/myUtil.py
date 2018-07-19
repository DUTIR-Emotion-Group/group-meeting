# -*- coding:utf-8 -*-
from numpy import *

# 数据文件转矩阵
# path： 数据文件路径
# delimiter: 行内字段分隔符
def file2matrix(path, delimiter):
    fp = open(path, "r")   # 读取文件内容
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()   # 按行转换为一维表
    # 逐行遍历，结果按分隔符分隔为行向量
    recordlist = [list(map(eval, row.split(delimiter))) for row in rowlist if row.strip()]
    # 返回转换后的矩阵形式
    return mat(recordlist)

# 随机生成聚类中心
def randCenters(dataSet, k):
    n = shape(dataSet)[1]   # 列数
    clustercents = mat(zeros((k, n)))   # 初始化聚类中心矩阵：k*n
    for col in range(n):
        mincol = min(dataSet[:, col])
        maxcol = max(dataSet[:, col])
        # random.rand(k, 1):产生一个0~1之间的随机数向量（k,1表示产生k行1列的随机数）
        clustercents[:, col] = mat(mincol + float(maxcol - mincol) * random.rand(k, 1))   # 按列赋值
    return clustercents

# 欧式距离计算公式
def distEclud(vecA, vecB):
    return linalg.norm(vecA-vecB)

# 绘制散点图
def drawScatter(plt, mydata, size=20, color='blue', mrkr='o'):
    plt.scatter(mydata.T[0].tolist(), mydata.T[1].tolist(), s=size, c=color, marker=mrkr)

# 以不同颜色绘制数据集里的点
def color_cluster(dataindx, dataSet, plt):
    datalen = len(dataindx)
    for indx in range(datalen):
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='blue', marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='green', marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='black', marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], c='cyan', marker='o')