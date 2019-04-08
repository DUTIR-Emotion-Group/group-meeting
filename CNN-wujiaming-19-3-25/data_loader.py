import numpy as np
import pandas as pd


def pre_deal():
    vacab_size = 10000
    pos_path = r'D:\data\GithubProject\cnn-text-classification-tf-master\data\rt-polaritydata\rt-polarity.pos'
    neg_path = r'D:\data\GithubProject\cnn-text-classification-tf-master\data\rt-polaritydata\rt-polarity.neg'


    with open(pos_path, 'r', encoding='utf-8') as pos_open:
        pos_lines = pos_open.readlines()
    with open(neg_path, 'r', encoding='utf-8') as neg_open:
        neg_lines = neg_open.readlines()

    df = pd.DataFrame({'sentences': pos_lines, 'label': [1] * len(pos_lines)})
    df = df.append(pd.DataFrame({'sentences': neg_lines, 'label': [0] * len(neg_lines)}), ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)


    freq_dct = {}
    for sent in df['sentences']:
        for word in sent.split(' '):
            if len(word) != 0:
                freq_dct[word] = freq_dct.get(word, 0) + 1

    print('total words: %d' % len(freq_dct))

    sorted_lst = sorted(freq_dct.items(), key=lambda x: x[1], reverse=True)

    word2index = {'\pad': 0}
    for i in range(vacab_size-1):
        word2index[sorted_lst[i][0]] = len(word2index)
    print('using most frequent words: %d' % len(word2index))

    df['id'] = [[word2index[word] if word in word2index
                 else 0 for word in sent.split(' ')] for sent in df['sentences']]

    print(df.head())
    return df

