from Tool.read_data import read_data
from Tool.word2vec_helper import read_word2vec
import jieba
#本文件对原始的数据集以及使用的词向量进行读取，然后对数据集进行分词，并和词向量进行对应，以及对未登录词随机初始化
#最终将处理后生成的一个与数据集相关的较小的词向量矩阵embedding_matrix以pickle文件进行存储，这样再次读取比较快，预处理后的数据集也对应保存文件
path = r"C:\Users\Administrator\Desktop\sgns.renmin.word"
embedding_matrix, word_list, V_size, embedding_dim = read_word2vec(path)
positive_list,negative_list = read_data()

corpus_word_set = set()
label_list = []
sentence_list = []
max_length = 256

#处理正例数据集
for pos in positive_list:
    tokens = " ".join(jieba.cut(pos.strip())).split(" ")
    label_list.append(1)
    sentence_list.append(tokens)
    for token in tokens:
        corpus_word_set.add(token)

#处理负例数据集
for neg in negative_list:
    tokens = " ".join(jieba.cut(neg.strip())).split(" ")
    label_list.append(-1)
    sentence_list.append(tokens)
    for token in tokens:
        corpus_word_set.add(token)

word_set = set(word_list)

import numpy as np
little_embedding_matrix = []
little_word_list = []
#embedding_matrix的第一个位置加入padding对应的零向量
little_embedding_matrix.append(np.zeros((300),dtype=np.float32))
little_word_list.append("padding_position")

intersection_set = word_set & corpus_word_set
out_of_vac_set = corpus_word_set - word_set

for word in intersection_set:
    little_word_list.append(word)
    little_embedding_matrix.append(embedding_matrix[word_list.index(word)])

for word in out_of_vac_set:
    little_word_list.append(word)
    little_embedding_matrix.append(np.random.uniform(-0.1,0.1,(300)))

little_embedding_matrix = np.asarray(little_embedding_matrix,dtype=np.float32)

import pickle
with open("../preprocess/little_embedding_matrix","wb") as file:
    pickle.dump(little_embedding_matrix,file)
    pickle.dump(little_word_list,file)

with open("../preprocess/sentence_with_word_id.txt","w") as file:
    for tokens,lable in zip(sentence_list,label_list):
        file.write(" ".join(tokens) + "\n")
        padding_list = ["0" for i in range(256)]
        for index, token in enumerate(tokens):
            padding_list[index] = str(little_word_list.index(token))
            if index == max_length - 1:
                break
        file.write(" ".join(padding_list))
        file.write("\n")
        file.write(str(lable)+"\n")



