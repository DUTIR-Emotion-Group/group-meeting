import re
import numpy as np


class MyLogisticRegression:
    def __init__(self):
        self.batch_size = 10
        self.epoch = 100
        self.learning_rate = 0.1

    def fit(self, x, y):
        self.data_x = np.asarray(x, np.float32)
        self.data_y = np.asarray(y, np.int)
        self.data_set_size = self.data_x.shape[0]

        self.weights = np.zeros(self.data_x.shape[1], dtype=np.float32)
        self.bias = np.array([0], dtype=np.float32)

        for round in range(self.epoch):
            self.data_pointer = 0
            batch_x, batch_y = self.get_next_batch()
            while batch_x is not None:
                # 计算batch上的预测概率  1.点乘 2.在第1维度进行累加 3.加上偏置 4.sigmoid
                pre_y = self.sigmoid(np.sum(batch_x * self.weights, axis=1) + self.bias)
                # 计算损失函数
                loss = np.mean(-(batch_y * np.log(pre_y) + (1 - batch_y) * np.log(1 - pre_y)))
                # 计算梯度
                gradient_w = - np.mean(((batch_y - pre_y) * batch_x.T).T, axis=0)
                gradient_b = - np.mean(batch_y - pre_y)
                # 更新参数
                self.weights -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b

                batch_x, batch_y = self.get_next_batch()

    def score(self, x, y):
        self.data_x = x
        self.data_y = y
        self.data_set_size = self.data_x.shape[0]

        self.pre_y = self.sigmoid(np.sum(self.data_x * self.weights, axis=1) + self.bias)
        loss = np.mean(- (self.data_y * np.log(self.pre_y) + (1 - self.data_y) * np.log(1 - self.pre_y)))
        return self.get_metirc()

    def predict(self, x):
        self.data_x = x
        pre_y = self.sigmoid(np.sum(self.data_x * self.weights, axis=1) + self.bias)
        return pre_y

    def get_next_batch(self):
        if self.data_pointer < self.data_set_size:
            next_pointer = self.data_pointer + self.batch_size
            if next_pointer < self.data_set_size:
                batch_x = self.data_x[self.data_pointer:next_pointer, :]
                batch_y = self.data_y[self.data_pointer:next_pointer]
            else:
                batch_x = self.data_x[self.data_pointer:, :]
                batch_y = self.data_y[self.data_pointer:]
            self.data_pointer = next_pointer
        else:
            batch_x = None
            batch_y = None
        return batch_x, batch_y

    def get_metirc(self):
        tp, tn, fp, fn = 0, 0, 0, 0
        for pre, label in zip(self.pre_y, self.data_y):
            if pre >= 0.5:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if label == 1:
                    fn += 1
                else:
                    tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        return precision, recall, f1, accuracy

    def sigmoid(self, x):
        x = np.asarray(x, dtype=np.float32)
        return 1 / (1 + np.exp(x * -1))

from sklearn.model_selection import KFold
x, y = [], []
with open(".\\data.txt","r")as file:
    lines = file.read().split("\n")
    for line in lines:
        triple = re.split(" +", line.strip())
        x.append(triple[0:2])
        y.append(triple[2])
x = np.asarray(x, dtype=np.float32)
y = np.asarray(y, dtype=np.int)

#5折交叉验证
fold_num = 5
kf = KFold(n_splits=fold_num,shuffle=True)
kf.get_n_splits(x)
p_arr_mLR, r_arr_mLR, f_arr_mLR, a_arr_mLR = np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32)
p_arr_oLR, r_arr_oLR, f_arr_oLR, a_arr_oLR = np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32),np.zeros(fold_num,dtype=np.float32)
fold = 0
for train_index,test_index in kf.split(x):
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]

    print("=================================== Running My Logistic Regression ===================================")
    LR = MyLogisticRegression()
    LR.fit(train_x,train_y)
    precision, recall, f1, accuracy = LR.score(test_x, test_y)

    p_arr_mLR[fold] = precision
    r_arr_mLR[fold] = recall
    f_arr_mLR[fold] = f1
    a_arr_mLR[fold] = accuracy
    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("f1:" + str(f1))
    print("accracy:" + str(accuracy))

    print("=================================== Running sk-learn Logistic Regression ===================================")
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn import metrics
    lr = LogisticRegression()
    lr.fit(train_x,train_y)
    pre_y = lr.predict(test_x)

    precision = metrics.precision_score(test_y,pre_y)
    recall = metrics.recall_score(test_y,pre_y)
    f1 = metrics.f1_score(test_y,pre_y)
    accuracy = metrics.accuracy_score(test_y,pre_y)
    
    p_arr_oLR[fold] = precision  
    r_arr_oLR[fold] = recall  
    f_arr_oLR[fold] = f1  
    a_arr_oLR[fold] = accuracy
    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("f1:" + str(f1))
    print("accracy:" + str(accuracy))

    fold += 1

print("My Logistic Regression fold validation result")

print("precision:" + str(round(np.mean(p_arr_mLR),4)) + "\t5-fold:" + str(p_arr_mLR))
print("recall:" + str(round(np.mean(r_arr_mLR),4)) + "\t5-fold:" + str(r_arr_mLR))
print("f1:" + str(round(np.mean(f_arr_mLR),4)) + "\t5-fold:" + str(f_arr_mLR))
print("accuracy:" + str(round(np.mean(a_arr_mLR),4)) + "\t5-fold:" + str(a_arr_mLR))

print("Official Logistic Regression fold validation result")
print("precision:" + str(round(np.mean(p_arr_oLR),4)) + "\t5-fold:" + str(p_arr_oLR))
print("recall:" + str(round(np.mean(r_arr_oLR),4)) + "\t5-fold:" + str(r_arr_oLR))
print("f1:" + str(round(np.mean(f_arr_oLR),4)) + "\t5-fold:" + str(f_arr_oLR))
print("accuracy:" + str(round(np.mean(a_arr_oLR),4)) + "\t5-fold:" + str(a_arr_oLR))