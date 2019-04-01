import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def get_embedding_matrix():
    with open(os.path.join(PROJECT_ROOT,"preprocess","little_embedding_matrix"),"rb") as file:
        embedding_matrix = pickle.load(file)
        word_list = pickle.load(file)
    return embedding_matrix, word_list

def get_train_test_set():
    with open(os.path.join(PROJECT_ROOT,"preprocess","sentence_with_word_id.txt"), "r") as file:
        word_ids_list = []
        label_list = []
        line = file.readline()
        while line:
            word_ids_list.append(file.readline().strip().split(" "))
            if int(file.readline().strip())==1:
                label_list.append([1,0])
            else:
                label_list.append([0,1])
            line = file.readline()
    word_ids_list = np.asarray(word_ids_list,dtype=np.int32)
    label_list = np.asarray(label_list,dtype=np.int32)
    #训练集70% 测试集30%
    return train_test_split(word_ids_list,label_list,test_size=0.3,random_state=1)