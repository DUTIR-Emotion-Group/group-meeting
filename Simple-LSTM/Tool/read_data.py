#读取原始的数据集
def read_data():
    positive_list = list(set([line[5:].strip() for line in open(r"..\data\pos.txt","r",encoding='utf8').readlines()]))
    negative_list = list(set([line[5:].strip() for line in open(r"..\data\neg.txt", "r", encoding='utf8').readlines()]))
    return positive_list,negative_list



