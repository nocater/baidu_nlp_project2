import pandas as pd
import os
from bert import tokenization
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bert")))
# import tokenization

dataset = 'baidu_95'
DATA_OUTPUT_DIR = './data/all_labels'
vocab_file = '../pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

# Load data
df = pd.read_csv(f'./{dataset}.csv', header=None, names=["labels", "item"], dtype=str)

# Create files
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "train"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "valid"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "test"))

# Split dataset
df_train = df[:int(len(df)*0.8)]
df_valid = df[int(len(df)*0.8):int(len(df)*0.9)]
df_test = df[int(len(df)*0.9):]

# Save dataset
file_set_type_list = ["train", "valid", "test"]
for file_set_type, df_data in zip(file_set_type_list, [df_train, df_valid, df_test]):
    predicate_out_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "predicate_out.txt"), "w",
                                        encoding='utf-8')
    text_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "text.txt"), "w",
                                        encoding='utf-8')
    token_in_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "token_in.txt"), "w",
                                        encoding='utf-8')
    token_in_not_UNK_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "token_in_not_UNK.txt"), "w",
                                        encoding='utf-8')

    # Processing
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)  # 初始化 bert_token 工具
    # feature
    text = '\n'.join(df_data.item)
    text_tokened = df_data.item.apply(bert_tokenizer.tokenize)
    text_tokened = '\n'.join([' '.join(row) for row in text_tokened])
    text_tokened_not_UNK = df_data.item.apply(bert_tokenizer.tokenize_not_UNK)
    text_tokened_not_UNK = '\n'.join([' '.join(row) for row in text_tokened_not_UNK])
    # label only choose first 3 lables: 高中 学科 一级知识点
    # if you want all labels
    # just remove list slice
    predicate_list = df_data.labels.apply(lambda x: x.split())
    predicate_list_str = '\n'.join([' '.join(row) for row in predicate_list])

    print(f'datasize: {len(df_data)}')
    text_f.write(text)
    token_in_f.write(text_tokened)
    token_in_not_UNK_f.write(text_tokened_not_UNK)
    predicate_out_f.write(predicate_list_str)

    text_f.close()
    token_in_f.close()
    token_in_not_UNK_f.close()
    predicate_out_f.close()