class Baidu_95_Multi_Label_Classification_Processor(DataProcessor):
    """Processor for the Baidu_95 data set"""
    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "predicate_out.txt"), encoding='utf-8') as predicate_out_f:
                token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                predicate_label_list = [seq.replace("\n", '') for seq in predicate_out_f.readlines()]
                assert len(token_in_list) == len(predicate_label_list)
                examples = list(zip(token_in_list, predicate_label_list))
                return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "test")), "test")

    def get_labels(self):
        """
        21 labels and 95 labels
        :return:
        """
        return ['生物技术实践', '公民道德与伦理常识', '经济学常识', '生活中的法律常识', '现代生物技术专题', '科学社会主义常识',
                '地球与地图', '生物', '近代史', '政治', '遗传与进化', '生物科学与社会', '分子与细胞', '人口与城市', '历史',
                '高中', '地理', '宇宙中的地球', '生产活动与地域联系', '古代史', '现代史', '稳态与环境']
        # return ['生物性污染', '细胞有丝分裂不同时期的特点', '液泡的结构和功能', '组成细胞的化学元素', '兴奋在神经纤维上的传导',
        #         '不完全显性', '免疫系统的组成', '生物技术在其他方面的应用', '群落的结构', '中央官制——三公九卿制', '核糖体的结构和功能',
        #         '人体免疫系统在维持稳态中的作用', '皇帝制度', '激素调节', '伴性遗传', '地球运动的地理意义', '宇宙中的地球', '地球运动的基本形式',
        #         '基因工程的原理及技术', '体液免疫的概念和过程', '基因的分离规律的实质及应用', '蛋白质的合成', '地球的内部圈层结构及特点',
        #         '人口增长与人口问题', '经济学常识', '劳动就业与守法经营', '器官移植', '生物技术实践', '垄断组织的出现', '基因工程的概念',
        #         '神经调节和体液调节的比较', '人口与城市', '组成细胞的化合物', '地理', '文艺的春天', '生物工程技术',
        #         '基因的自由组合规律的实质及应用', '郡县制', '人体水盐平衡调节', '内质网的结构和功能', '人体的体温调节',
        #         '免疫系统的功能', '科学社会主义常识', '与细胞分裂有关的细胞器', '太阳对地球的影响', '古代史', '清末民主革命风潮',
        #         '复等位基因', '人工授精、试管婴儿等生殖技术', '“重农抑商”政策', '生态系统的营养结构', '减数分裂的概念',
        #         '地球的外部圈层结构及特点', '细胞的多样性和统一性', '政治', '工业区位因素', '细胞大小与物质运输的关系',
        #         '夏商两代的政治制度', '农业区位因素', '溶酶体的结构和功能', '生产活动与地域联系', '内环境的稳态', '遗传与进化',
        #         '胚胎移植', '生物科学与社会', '近代史', '第三产业的兴起和“新经济”的出现', '公民道德与伦理常识', '中心体的结构和功能',
        #         '社会主义市场经济的伦理要求', '高中', '选官、用官制度的变化', '减数分裂与有丝分裂的比较', '遗传的细胞基础',
        #         '地球所处的宇宙环境', '培养基与无菌技术', '生活中的法律常识', '高尔基体的结构和功能', '社会主义是中国人民的历史性选择',
        #         '人口迁移与人口流动', '现代史', '地球与地图', '走进细胞', '生物', '避孕的原理和方法', '血糖平衡的调节',
        #         '现代生物技术专题', '海峡两岸关系的发展', '生命活动离不开细胞', '兴奋在神经元之间的传递', '历史', '分子与细胞',
        #         '拉马克的进化学说', '遗传的分子基础', '稳态与环境']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_str = line[0]
            predicate_label_str = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_str, text_b=None, label=predicate_label_str))
        return examples