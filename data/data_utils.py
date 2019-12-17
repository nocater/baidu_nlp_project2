import pandas as pd
import os
from collections import Counter
import numpy as np

def data2file(file_dir, type='csv'):
    """
    带知识点数据处理
    :param file_dir: 原始数据集文件路径
    :param type: 数据类型 csv or fasttest
    :return:
    """
    grades = ['高中']
    subjects = ['地理', '历史', '生物', '政治']
    categories = {  '地理':['地球与地图','宇宙中的地球','生产活动与地域联系','人口与城市','区域可持续发展'],
                    '历史':['古代史', '近代史', '现代史'],
                    '生物':['现代生物技术专题','生物科学与社会','生物技术实践','稳态与环境','遗传与进化','分子与细胞'],
                    '政治':['经济学常识', '科学思维常识', '生活中的法律常识','科学社会主义常识','公民道德与伦理常识','时事政治']
                  }

    df_target = pd.DataFrame(columns=['labels','item'])
    for grade in grades:
        for subject in subjects:
            for category in categories[subject]:
                file = os.path.join(file_dir, grade+'_'+subject, 'origin', category+'.csv')
                df = pd.read_csv(open(file, encoding='utf8'))
                print(f'{grade} {subject} {category} \tsize:{len(df)}')

                # 按网页顺序对其排序
                df['web-scraper-order'] = df['web-scraper-order'].apply(lambda x: int(x.split('-')[1]))
                df = df[['web-scraper-order','item']]
                df = df.sort_values(by='web-scraper-order')

                # 删除文本中的换行
                df['item'] = df.item.apply(lambda x:  "".join(x.split()))
                df['labels'] = df.item.apply(lambda x: [grade, subject, category]+x[x.index('[知识点：]')+6:].split(',') if x.find('[知识点：]')!=-1 else [grade, subject, category])
                df['item'] = df.item.apply(lambda x:  x.replace('[题目]',''))
                df['item'] = df.item.apply(lambda x:  x[:x.index('题型')] if x.index('题型') else x)

                df = df[['labels','item']]
                df_target = df_target.append(df)

    print('origin data size:', len(df_target))
    print(df_target.head())

    # 设置样本数量阈值
    min_samples = 300
    # 阈值 标签数
    # 500   64
    # 400   75
    # 300   95
    # 200   134
    # 100   228

    df = df_target.copy()
    labels = []
    for i in df.labels:
        labels.extend(i)

    result = dict(sorted(dict(Counter(labels)).items(), key=lambda x: x[1], reverse=True))
    lens = np.array(list(result.values()))
    LABEL_NUM = len(lens[lens > min_samples])

    # 选定数据label
    label_target = set([k for k, v in result.items() if v > min_samples])

    #
    df['labels'] = df.labels.apply(
        lambda x: x[:3] + list(set(x) - set(x[:3]) & label_target))  # 保证 grade subject category 在前三位置
    df['labels'] = df.labels.apply(lambda x: None if len(x) < 4 else x)  # 去除没有知识点的数据
    df = df[df.labels.notna()]

    # 最终的labels数量
    labels = []
    [labels.extend(i) for i in df.labels]
    LABEL_NUM = len(set(labels))

    print(f'>{min_samples} datasize:{len(df)} multi_class:{LABEL_NUM}')

    if type=='csv':
        # save
        profix = ''

        if profix:
            df['labels'] = df.labels.apply(lambda x: [profix + i for i in x])

        df['labels'] = df.labels.apply(lambda x: ' '.join(x))

        # shuffle
        df = df.sample(frac=1)

        file = os.path.join(file_dir, f'baidu_{LABEL_NUM}{profix}.csv')

        df.to_csv(file, index=False, sep=',', header=False, encoding='UTF8')  # 当sep 字符在df中存在会在字符串前后添加引号
        print('csv data file generated! ', file)
    elif type=='fasttest':
        # save
        profix = '__label__'

        if profix:
            df['labels'] = df.labels.apply(lambda x: [profix + i for i in x])

        df['labels'] = df.labels.apply(lambda x: ' '.join(x))

        # shuffle
        df = df.sample(frac=1)

        file = os.path.join(file_dir, f'baidu_{LABEL_NUM}{profix}.csv')

        with open(file, 'w') as f:
            for index, row in df.iterrows():
                f.write(row['labels'] + ' ' + row['item'] + '\n')
        print('fasttext data file generated! ', file)
    else:
        print('Error Type!')


def data2file_without_knowledge(file_dir):
    grades = ['高中']
    subjects = ['地理', '历史', '生物', '政治']
    categories = {'地理': ['地球与地图', '宇宙中的地球', '生产活动与地域联系', '人口与城市', '区域可持续发展'],
                  '历史': ['古代史', '近代史', '现代史'],
                  '生物': ['现代生物技术专题', '生物科学与社会', '生物技术实践', '稳态与环境', '遗传与进化', '分子与细胞'],
                  '政治': ['经济学常识', '科学思维常识', '生活中的法律常识', '科学社会主义常识', '公民道德与伦理常识', '时事政治']
                  }

    for grade in grades:
        for subject in subjects:
            for category in categories[subject]:
                file = os.path.join(file_dir, f'{grade}_{subject}', 'origin', category+'.csv')
                df = pd.read_csv(open(file, encoding='utf8'))
                print('size:', len(df))

                # 按网页顺序对其排序
                df['web-scraper-order'] = df['web-scraper-order'].apply(lambda x: int(x.split('-')[1]))
                df = df[['web-scraper-order', 'item']]
                df = df.sort_values(by='web-scraper-order')

                # 对文本处理
                def foo(x):
                    x = x.strip()
                    x = x.replace('\n', '')
                    x = x.replace('\t', '')
                    x = x.replace('\r', '')
                    x = x.replace(' ', '')
                    x = x.replace(' ', '')
                    x = x[:x.index('题型')]
                    return x

                # 删除文本中的换行
                df['item'] = df.item.apply(lambda x: foo(x))
                df = df['item']

                # save
                with open(os.path.join(file_dir, f'{grade}_{subject}', category+'_test.csv'), 'w', encoding='utf8') as f:
                    f.write('\n'.join(list(df.values)))
                category + 'Done!'


if __name__ == '__main__':
    data2file_without_knowledge(r'D:\Dataset\百度题库')