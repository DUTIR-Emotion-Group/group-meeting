# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

'''
语料预处理程序
函数：均在main中使用
def build_tokenizer(fnames, max_seq_len, dat_fname) 建立语料的序列化文件
def _load_word_vec(path, word2idx=None) 加载预训练词向量
def build_embedding_matrix(word2idx, embed_dim, dat_fname) 建立针对语料的emb权重矩阵

类：
序列化类
class Tokenizer(object)
    def __init__(self, max_seq_len, lower=True)
    def fit_on_text(self, text)

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0.)

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post')

class ABSADataset(Dataset)
    def __init__(self, fname, tokenizer)
    def __getitem__(self, index)
    def __len__(self)
'''

'''
@description: 建立语料的序列化文件
@param {原始语料文件名列表 序列最大长度 序列化文件名} 
@return: 保存语料的序列化结果，并返回该结果
'''
def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname): # 如果语料的序列化文件已经存在
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else: # 如果语料的序列化文件不存在
        text = ''
        for fname in fnames:
            # newline用于指定换行符
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3): # 步长为3
                # partition:返回一个3元组，第一个为分隔符左边的子串，第二个为分隔符本身，第三个为分隔符右边的子串
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len) # Tokenizer是语料的序列化类
        tokenizer.fit_on_text(text) # 构建word-id对应关系，是两个字典
        pickle.dump(tokenizer, open(dat_fname, 'wb')) # 保存语料的序列化文件
    return tokenizer # 返回Tokenizer实例

'''
@description: 使用预训练词向量，为语料建立word2vec字典
@param {预训练词向量文件名 语料中的word-id对} 
@return: word2vec字典
'''
def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split() # rstrip()，删除string末尾的指定字符（默认为空格）
        # 如果预训练词向量的包含的单词在语料中出现，就把单词及对应向量添加到字典中
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec

'''
@description: 根据预训练词向量建立针对语料的词向量矩阵
@param {语料中的word-id对 向量维度 使用预训练词向量初始化的语料词向量文件} 
@return: 保存初始化完成的语料词向量矩阵，并返回该矩阵
'''
def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    # 如果该语料的词向量文件已经存在
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        # fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
        #     if embed_dim != 300 else './glove.42B.300d.txt'
        if embed_dim != 300:
            fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt'
        else:
            fname = './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            # 若果单词有对应的词向量
            if vec is not None:
                # embedding_matrix[0]以及embedding_matrix[len(word2idx)+1]为全0
                # 单词的id与单词存储在矩阵中的行序号一致，当然理解这个似乎没什么用
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

'''
@description: 建立转换字典->序列数字化->截断、补零
@param {type} 
@return: 数字化序列
'''
class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    # 建立转换字典
    # 分词、小写、建立word2idx、idx2word
    # 序号从1开始递增
    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod # https://blog.csdn.net/weixin_41010198/article/details/84828022 解释很好
    # 句子数字序列的截断和补齐函数
    def pad_sequence(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype) # x初始是全0的序列，长度是maxlen
        # 截断
        if truncating == 'pre': # 向前
            trunc = sequence[-maxlen:]
        else: # 向后
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype) # 将数据转化为ndarray类型
        # 对齐
        if padding == 'post': # 在序列后面补零
            x[:len(trunc)] = trunc
        else: # 在序列前面补零
            x[-len(trunc):] = trunc
        return x

    # 数字化
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1 # unk的索引是len+1，因为idx是从1开始的
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = [] # 每个元素是一个data字典，定义见下
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)

            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)

            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0) # 方面左侧的文本长度
            aspect_len = np.sum(aspect_indices != 0) # 方面的单词长度
            # 求得方面在sequence中位置
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)