# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:57
import pandas as pd


# 读取txt文件
def read_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines()]
    # labels, texts = [], []
    list=[]
    for line in content:
        parts = line.split('\t')
        label, text = parts[0], ''.join(parts[1:])
        # labels.append(label)
        # texts.append(text)
        list.append((label,text))

    return list


file_path = 'data/train.tsv'
train_df=read_csv_file(file_path)
# labels, texts = read_csv_file(file_path)
# train_df = pd.DataFrame({'label': labels, 'text': texts})

file_path = 'data/dev.tsv'
test_df=read_csv_file(file_path)
# labels, texts = read_csv_file(file_path)
# test_df = pd.DataFrame({'label': labels, 'text': texts})

# print(train_df.head())
# print(test_df.head())
#
# train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
# print(train_df.describe())

