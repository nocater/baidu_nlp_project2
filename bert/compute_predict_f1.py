import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)
    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return micro_f1, macro_f1


mlb = MultiLabelBinarizer()
all_labels = ['生物性污染', '细胞有丝分裂不同时期的特点', '液泡的结构和功能', '组成细胞的化学元素', '兴奋在神经纤维上的传导',
                '不完全显性', '免疫系统的组成', '生物技术在其他方面的应用', '群落的结构', '中央官制——三公九卿制', '核糖体的结构和功能',
                '人体免疫系统在维持稳态中的作用', '皇帝制度', '激素调节', '伴性遗传', '地球运动的地理意义', '宇宙中的地球', '地球运动的基本形式',
                '基因工程的原理及技术', '体液免疫的概念和过程', '基因的分离规律的实质及应用', '蛋白质的合成', '地球的内部圈层结构及特点',
                '人口增长与人口问题', '经济学常识', '劳动就业与守法经营', '器官移植', '生物技术实践', '垄断组织的出现', '基因工程的概念',
                '神经调节和体液调节的比较', '人口与城市', '组成细胞的化合物', '地理', '文艺的春天', '生物工程技术',
                '基因的自由组合规律的实质及应用', '郡县制', '人体水盐平衡调节', '内质网的结构和功能', '人体的体温调节',
                '免疫系统的功能', '科学社会主义常识', '与细胞分裂有关的细胞器', '太阳对地球的影响', '古代史', '清末民主革命风潮',
                '复等位基因', '人工授精、试管婴儿等生殖技术', '“重农抑商”政策', '生态系统的营养结构', '减数分裂的概念',
                '地球的外部圈层结构及特点', '细胞的多样性和统一性', '政治', '工业区位因素', '细胞大小与物质运输的关系',
                '夏商两代的政治制度', '农业区位因素', '溶酶体的结构和功能', '生产活动与地域联系', '内环境的稳态', '遗传与进化',
                '胚胎移植', '生物科学与社会', '近代史', '第三产业的兴起和“新经济”的出现', '公民道德与伦理常识', '中心体的结构和功能',
                '社会主义市场经济的伦理要求', '高中', '选官、用官制度的变化', '减数分裂与有丝分裂的比较', '遗传的细胞基础',
                '地球所处的宇宙环境', '培养基与无菌技术', '生活中的法律常识', '高尔基体的结构和功能', '社会主义是中国人民的历史性选择',
                '人口迁移与人口流动', '现代史', '地球与地图', '走进细胞', '生物', '避孕的原理和方法', '血糖平衡的调节',
                '现代生物技术专题', '海峡两岸关系的发展', '生命活动离不开细胞', '兴奋在神经元之间的传递', '历史', '分子与细胞',
                '拉马克的进化学说', '遗传的分子基础', '稳态与环境']

mlb.fit([[label] for label in all_labels])

true_file = './data/all_labels/test/predicate_out.txt'
predcit_file = '../output/predicate_classification_model/epochs6_baidu_95_all/predicate_predict.txt'

y_true, y_pred = [], []
with open(true_file, encoding='utf8') as f:
    for line in f.readlines():
        y_true.append(line.split())

with open(predcit_file, encoding='utf8') as f:
    for line in f.readlines():
        y_pred.append(line.split())

y_true = mlb.transform(y_true)
y_pred = mlb.transform(y_pred)

print('micro_f1, macro_f1:', f1_np(y_true, y_pred))