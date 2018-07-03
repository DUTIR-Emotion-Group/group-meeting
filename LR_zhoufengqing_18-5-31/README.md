# Logistic Regression 代码简单实现

简单实现了一个MyLogisticRegression类<br>
其中可设置参数包括<br>
batch_size  每次训练批数据大小<br>
epoch 数据训练轮数<br>
learning_rate 学习率<br>
<br>
训练中使用的损失是交叉熵<br>
优化方法为梯度下降法<br>
数据集来自网络，总共100组数据，二分类，特征维度为2<br>
训练集和测试机的比例为7:3，随机进行划分<br>
<br>
实验结果如下：<br>
precision:0.846153846153846<br>
recall:0.9166666666666666<br>
f1:0.8799999999999999<br>
accracy:0.9<br>
可视化结果如下：<br>

![image](https://github.com/DUTIR-Emotion-Group/group-meeting/blob/master/LR_zhoufengqing_18-5-31/img/myLR_result.png)


下面是调用sk-learn中实现的LogisticRegression模型进行的实验<br>
实验结果如下：<br>
precision:1.0<br>
recall:0.9166666666666666<br>
f1:0.9565217391304348<br>
accracy:0.9666666666666667<br>
可视化结果如下：<br>

![image](https://github.com/DUTIR-Emotion-Group/group-meeting/blob/master/LR_zhoufengqing_18-5-31/img/officialLR_result.png)

分析：<br>
1.由于自己实现的代码没有进行参数调节(batch_size=10,epoch=100,learning_rate = 0.1),各个p、r、f指标都略低于sk-learning中默认参数的LR模型，
通过一定的参数调整，结果能够得到提升。<br>
2.从两幅模型结果可视化观察可知，sk-learn中的LR模型的分类边界明显清晰，而本人自己实现的LR模型由于参数优化方面不足，导致训练出来的模型对一些处在分类边界附近的数据区分能力较弱，导致图中的边界模糊不清晰。<br>

