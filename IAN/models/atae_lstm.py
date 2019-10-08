# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)

        x = self.embed(text_raw_indices) # (128, 80, 300) (batch_size, max_seq_len, emb_dim)
        x = self.squeeze_embedding(x, x_len) # (128, 32, 300)，压缩max_seq_len到实际最大句子长度
        aspect = self.embed(aspect_indices) # (128, 80, 300)
        # 前项求每个aspect各个词向量之和(128, 300)，后项求词向量个数(128, 1)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1)) # (128, 300)
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1) # (128, 32, 300)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len) # (128, 32, 300) (batch_size, x_len_max, hid_dim)
        ha = torch.cat((h, aspect), dim=-1) # (128, 32, 600)
        _, score = self.attention(ha) # score(128, 1, 32)
        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2
        output = torch.squeeze(torch.bmm(score, h), dim=1) # (128, 300)

        out = self.dense(output)
        return out