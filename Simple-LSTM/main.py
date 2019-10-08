from model.LSTM import LSTM
from Tool.get_preprocess_data import *

train_x,test_x,train_y,test_y = get_train_test_set()
embedding_matrix,word_list = get_embedding_matrix()
lstm = LSTM(embedding_matrix, train_x, train_y, test_x, test_y)
lstm.run()