# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:13:05 2018

@author: Administrator
"""

from numpy import *
def loadDataSet(fileName):  
    dataMat = []              
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float, curLine)) 
        dataMat.append(fltLine)
        #dataMat.append(curLine)
    return dataMat
 
# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) 

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids
 
# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离,1,2
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  
            clusterAssment[i,:] = minIndex,minDist**2   
        print (centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   
            centroids[cent,:] = mean(ptsInClust, axis = 0) 
    return centroids, clusterAssment


datMat = mat(loadDataSet('xigua.csv'))
myCentroids,clustAssing = kMeans(datMat,2)
print (myCentroids)
print (clustAssing)

import matplotlib.pyplot as plt
for i in range(len(datMat)):
    if int(clustAssing[i,0])==0:
        plt.scatter(datMat[i,0],datMat[i,1],color='red')
    if int(clustAssing[i,0])==1:
        plt.scatter(datMat[i,0],datMat[i,1],color='black')
#    if int(clustAssing[i,0])==2:
#       plt.scatter(datMat[i,0],datMat[i,1],color='blue')
#    if int(clustAssing[i,0])==3:
#        plt.scatter(datMat[i,0],datMat[i,1],color='yellow')