# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from sklearn import metrics
from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN

'''
class Instructor:
    def __init__(self, opt):
    def _print_args(self):
    def _reset_params(self):
    def _train(self, criterion, optimizer, max_test_acc_overall=0):
    def _evaluate_acc_f1(self):
    def run(self, repeats=1):

if __name__ == '__main__':
'''
class Instructor:
    def __init__(self, opt):
        self.opt = opt

        # 建立语料的序列化文件
        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        # 建立语料的词向量文件
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        # 建立训练集、测试集
        trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        # 建立训练集和测试集的迭代器
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
        # 选择模型、运行设备
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        if opt.device.type == 'cuda': # 独显内存分配情况
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)) # prod返回输入张量所有元素的积
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        max_test_acc = 0
        max_f1 = 0
        global_step = 0 # 统计在一次训练（多个epoch）中，总共处理的batch的个数
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            # enumerate将一个可遍历对象处理为(序号, 对象)的形式
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                # 模型的输入是序列化后的句子
                # 输入是一个batch的句子，输出是一个矩阵
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 每log_step个batch计算一次train_acc、test_acc、f1
                if global_step % self.opt.log_step == 0:
                    # torch.argmax()返回值最大的一维的维度序号，第二个参数指定在哪个维度进行比价
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1() # 测试集跑一跑
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        # 保存模型
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_{1}_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(test_acc, 4))
                            torch.save(self.model.state_dict(), path)
                            print('>> saved: ' + path)
                    if f1 > max_f1:
                        max_f1 = f1

                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', train_acc, global_step)
                    writer.add_scalar('test_acc', test_acc, global_step)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))

        writer.close()

        # 返回该epoch的最大test_acc，最大test_f1
        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            print('repeat: ', i)
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall=max_test_acc_overall)
            print('max_test_acc: {0}     max_f1: {1}'.format(max_test_acc, max_f1))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            print('#' * 100)
        print("max_test_acc_overall:", max_test_acc_overall)
        print("max_f1_overall:", max_f1_overall)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='atae_lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    # 为使每一层输出的方差尽可能相等，将参数在某个区间内平均初始化
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    # 命令行输入的时候是小写，模型类是大写
    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    # 妙啊
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    # 命令行可选参数：模型、语料、输入、初始化方法、优化方法
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run(5)