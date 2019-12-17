import os
import numpy as np


def baidu_3layers_data(file_dir):
    """
    将不带知识点的三层标签数据处理成多标签数据
    过滤数据小于500的数据
    :param file_dir:
    :return:
    """
    subjects = ['高中_地理', '高中_历史', '高中_生物', '高中_政治']
    label_id = dict(zip(subjects, range(4)))
    label_count = {}

    x = []
    y = []

    for label in subjects:
        for doc in os.listdir(file_dir + label):
            if not doc.endswith('csv'):
                continue

            sublabel = os.path.splitext(doc)[0]
            data = open(file_dir + label + '/' + doc, encoding='utf8').readlines()
            data = list(map(lambda x: x[4:], data))

            if len(data) < 500: continue

            x.extend(data)
            label_count[label + '_' + sublabel] = len(data)
            label_id[label + '_' + sublabel] = len(label_id)
            y.extend([[label_id[label], label_id[label + '_' + sublabel]]] * len(data))

    # 去除内容中的制表符
    x = [''.join(i.split()) for i in x]

    len(x), len(y), label_id, label_count, sum(label_count.values())
    with open('../data/kkb/baidu_ernie_x_19.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(x))
    with open('../data/kkb/baidu_ernie_y_19.txt', 'w', encoding='utf8') as f:
        f.write(repr(y))
    with open('../data/kkb/baidu_ernie_label2id_19.json', 'w', encoding='utf8') as f:
        f.write(repr(label_id))

    return x,y


def totsv(x, y):
    """
    将数据编码成ERNIE需要数据
    :param x:
    :param y:
    :return:
    """
    np.random.seed(1)
    indexs = list(range(len(x)))
    np.random.shuffle(indexs)
    x_, y_ = np.array(x)[indexs], np.array(y)[indexs]

    train_x, train_y = x_[:-4000], y_[:-4000]
    dev_x, dev_y = x_[-4000:-2000], y_[-4000:-2000]
    test_x, test_y = x_[-2000:], y_[-2000:]

    with open('./data/kkb/test_multi.tsv', 'w', encoding='utf8') as f:
        f.write('label\ttext_a\n')
        for (a, b) in zip(test_x, test_y):
            a = ''.join(a.split()) + '\n'
            f.write(str(list(b)) + '\t' + a)
            # f.write(repr(list(b))+'\t'+a)

    with open('./data/kkb/dev_multi.tsv', 'w', encoding='utf8') as f:
        f.write('label\ttext_a\n')
        for (a, b) in zip(dev_x, dev_y):
            a = ''.join(a.split()) + '\n'
            f.write(str(list(b)) + '\t' + a)
            # f.write(repr(list(b))+'\t'+a)

    with open('./data/kkb/train_multi.tsv', 'w', encoding='utf8') as f:
        f.write('label\ttext_a\n')
        i = 0
        for (a, b) in zip(train_x, train_y):
            a = ''.join(a.split()) + '\n'
            f.write(str(list(b)) + '\t' + a)
            # f.write(repr(list(b))+'\t'+a)


if __name__ == '__main__':
    x, y = baidu_3layers_data(r'D:\Dataset\百度题库\\')
    totsv(x, y)
