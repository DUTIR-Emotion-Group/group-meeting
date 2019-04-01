import numpy as np
#读取词向量函数，返回词向量矩阵以及一些相关的基本信息
def read_word2vec(path):
    with open(path,'r',encoding="utf8",errors="ignore")as file:
        line = file.readline()
        embedding_matrix=[]
        word_list = []
        if line != None and len(line.strip().split(" ")) == 2:
            # 词向量首行为词典大小以及词向量维度
            V_size = line.strip().split(" ")[0]
            embedding_dim = line.strip().split(" ")[1]
            line = file.readline()
        while line:
            line_splited = line.strip().split(" ")
            embedding_matrix.append(line_splited[1:])
            word_list.append(line_splited[0])
            line = file.readline()
        return np.asarray(embedding_matrix,dtype=np.float32), word_list, V_size, embedding_dim
