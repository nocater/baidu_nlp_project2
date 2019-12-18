# baidu_nlp_project2
开课吧&amp;后厂理工学院 百度NLP项目2：百度试题数据集多标签文本分类

# 1.数据说明
原始数据集为`高中`下`地理`,`历史`,`生物`,`政治`四门学科数据，每个学科下各包含第一层知识点，如`历史`下分为`近代史`,`现代史`,`古代史`。  
原始数据示例： 

> [题目]  
我国经济体制改革首先在农村展开。率先实行包产到组、包产到户的农业生产责任制的省份是（    ）  
①四川        ②广东        ③安徽       ④湖北A. ①③B. ①④C. ②④D. ②③题型: 单选题|难度: 简单|使用次数: 0|纠错复制收藏到空间加入选题篮查看答案解析答案：A解析：本题主要考察的是对知识的识记能力，比较容易。根据所学知识可知，在四川和安徽，率先实行包产到组、包产到户的农业生产责任制，故①③正确；②④不是。所以答案选A。知识点：  
[知识点：]  
经济体制改革,中国的振兴

对数据处理：
- 将数据的[知识点：]作为数据的第四层标签，显然不同数据的第四层标签数量不一致
- 仅保留题目作为数据特征，删除[题型]及[答案解析]

# 2.3层标签数据集
根据阈值(500,1000)对数据进行过滤，可以分类得到19类和13类两组数据，其中19类数据具有类别不平衡问题。  
**因比较简单，此问题未在课上讲解**

## 模型
1. bert_keras
利用`bert_keras`对原始数据进行多标签文本分类，变种包括：13类，19类，19类(处理类别不均衡)，19类&AWM等  
Arrange Word Matrix方法取自图神经网络方法:   
[Hierarchical Taxonomy-Aware and Attentional Graph Capsule RCNNs for Large-Scale Multi-Label Text Classiﬁcation](https://arxiv.org/abs/1906.04898)

2. ERNIE1.0
完成单分类，暂时放弃。待ERNIE2.0发布之后跟进。

# 3.4层标签数据集
## 模型
1. fasttest
2. textcnn
3. gcn  
  [GCN with Multi Labels](https://github.com/nocater/text_gcn)  
  [GCN_AAAI2019](https://github.com/yao8839836/text_gcn/) 
4. bert
5. xlnet(doing)

# 4.实验结果
|数据集|模型|类别|Acc|Micro-F1|Macro-F1|备注|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Baidu|ERNIE|2|0.73|-|-|single classify|
|Baidu|BERT|13|-|0.9299|0.8615|multi_labels classify 13|
|Baidu|BERT|19|-|0.8996 |0.6797|multi_labels classify 19|
|Biadu|FastText|19|-|0.42|0.21|multi_labels classify 19(imbalance)|
|Baidu|GCN-BERT|19|-|0.90|0.78|multi_labels classify 19(balance)|
|Baidu|GCN-BERT|19|-|0.89|0.69|multi_labels classify 19(imbalance)|
|Baidu|FastText|95|-|0.421|0.234|epoch 1000, ngram 5, dim 50|
|Baidu|TextCnn|95|-|0.00478|0.028|epoch 10, lr 0.005, padding 128|
|Baidu|GCN|95|-|0.8755|0.6914|gcn|
|Baidu|BERT|21|0.7958|0.941|0.163|BERT 3 layers labels result|
|Baidu|BERT|95|0.5788|0.917|0.781|only BERT|
