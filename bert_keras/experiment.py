#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

from bert_keras.arrange_word_matrix import *

import keras
keras.__version__


maxlen = 100
config_path = 'model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'model/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def load_data(file_x, file_y, n_class=19, imbalance=False, use_awm=False, fixed_nums=900):
    """
    加载数据
    :param file_x:
    :param file_y:
    :param n_class: 类别数
    :param imbalance: 是否处理列别不平衡
    :param use_awm: 是否使用Arrange Word Matrix
    :param fixed_nums:
    :return:
    """

    x, y = None, None
    # if n_class == 13:
    x = open(file_x, 'r', encoding='utf8').readlines()
    y = eval(open(file_y, 'r', encoding='utf8').readlines()[0])
    # elif n_class == 19:
    #     x = open('./data/kkb/x_19.txt', 'r', encoding='utf8').readlines()
    #     y = eval(open('./data/kkb/y_19.txt', 'r', encoding='utf8').readlines()[0])

    print('{} class task, data size:{},{}'.format(n_class, len(x), len(y)))

    # 类别不平衡
    if imbalance:
        ys = [str(i) for i in y]
        from collections import Counter
        counts = Counter(ys)
        for k, v in counts.items():
            print(k, ':', v)

        df = pd.DataFrame({'x': x, 'y': ys})
        new_x, new_y = [], []
        for k, v in counts.items():
            x_ = np.random.choice(df[df.y == k].x, fixed_nums)
            y_ = [eval(k)] * 900
            new_x.extend(x_)
            new_y.extend(y_)
        print('after deal data imbalance, data size:', len(new_x), len(new_y))
        x, y = new_x, new_y

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(range(n_class))
    y = mlb.fit_transform(y)
    y = [list(i) for i in y]

    # arrage word matrix
    x_awm = []
    if use_awm:
        for row in x[:10]:
            _, awm = text2matrix(row)
            x_awm.append(list(np.reshape(awm[:20], (-1))))
    x = x_awm

    data = list(zip(x, y))
    np.random.shuffle(data)
    train_data = data[:-4000]
    val_data = data[-4000:-2000]
    test_data = data[-2000:]

    return train_data, val_data, test_data


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32, multi_labels=False):
        self.data = data
        self.batch_size = batch_size
        self.multi_labels = multi_labels
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    if self.multi_labels: Y = Y.reshape(-1, np.shape(Y)[-1])
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def micro_f1(y_true, y_pred):
    """F1 metric.

    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))

    """Micro_F1 metric.
    """
    precision = K.sum(true_positives) / K.sum(predicted_positives)
    recall = K.sum(true_positives) / K.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return micro_f1


def macro_f1(y_true, y_pred):
    """F1 metric.

    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))

    """Micro_F1 metric.
    """
    precision = K.sum(true_positives) / K.sum(predicted_positives)
    recall = K.sum(true_positives) / K.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return macro_f1


def create_model(config_path, checkpoint_path):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(13, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    # val_metric = Metrics([val_x,val_y])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=[micro_f1, macro_f1]
    )
    model.summary()


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R


if __name__ == '__main__':
    train_data, val_data, test_data = load_data()

    # bert config parameters
    maxlen = 100
    config_path = 'model/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'model/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'model/chinese_L-12_H-768_A-12/vocab.txt'

    token_dict = {}

    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = OurTokenizer(token_dict)

    model = create_model()

    train_D = data_generator(train_data, multi_labels=True)
    valid_D = data_generator(val_data, multi_labels=True)
    test_D = data_generator(test_data, multi_labels=True)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=1,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        # callbacks=[val_metric],
    )

    test_D = data_generator(test_data, multi_labels=True)
    model.evaluate_generator(test_D.__iter__(), len(test_D))