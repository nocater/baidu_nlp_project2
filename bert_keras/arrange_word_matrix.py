import jieba
import numpy as np


def count_words(s):
    stop_words = ['$', '?', '_', '“', '”', '、', '。', '《', '》', '，', '（', '）', '\n', '的', '了', '是']
    tokenstr = []
    result = {}

    word2pos = {}
    pos2word = {}

    words = jieba.cut(s)

    i = 0
    for word in words:
        if word in stop_words: continue
        tokenstr.append(word)
        result[word] = result.get(word, 0) + 1
        pos2word[i] = word

        indexs = word2pos.get(word, [])
        indexs.append(i)
        word2pos[word] = indexs

        i += 1

    result = dict(sorted(result.items(), key=lambda x: (x[1], x[0]), reverse=True))
    wordslist = list(result.keys())
    assert len(set(tokenstr)) == len(wordslist)
    return (wordslist, tokenstr, word2pos, pos2word)


def fill_table(TD_list, related_tables, target_width, qqueue):
    TD_list[0] = qqueue[0]  # TD_list 长度为target_width 第一个位置对应此单词在wlist中的索引。0,1,2...
    count = 1

    while qqueue != [] and count < target_width:
        use_index = qqueue[0]  # 单词索引
        del qqueue[0]
        use_list = related_tables[use_index]  # 取出use_index单词对应的相关单词。
        len1 = len(use_list)  # 查看 i对应 的相关单词的个数。
        len2 = target_width - count
        if len1 >= len2:  # 大体意思应该是查看单词i对应的相关单词个数如果满足 target_width就直接从相关单词按顺序取出来填充到TD_list中。
            TD_list[count:] = use_list[:len2]
            assert len(TD_list) == target_width
            count = target_width
            break
        else:  # 如果不满足就有多少填多少。剩下的用 -1填充。
            TD_list[count:count + len1] = use_list
            assert len(TD_list) == target_width
            count += len1
            for next_id in use_list:
                qqueue.append(next_id)
    for i in range(count, target_width):
        TD_list[i] = -1


def reorder(table, word2pos, pos2word, wlist, word2id):
    sort_table = []
    topn, neighbor = np.array(table).shape
    for i in range(topn):
        tmp = []
        tmp += word2pos[wlist[table[i][0]]]  # record each center word index
        length = len(tmp)  # occurred times of center words
        t = []  # t is use to related words index
        for j in range(1, neighbor):
            t += word2pos[wlist[table[i][j]]]
        index = np.random.randint(len(t), size=20 - length)
        t = np.array(t)
        t = list(t[index])
        tmp = tmp + t  # conccat the index of center word and index of its related words
        tmp.sort()
        for j in range(len(tmp)):
            tmp[j] = word2id[pos2word[tmp[j]]]  # convert index to word_id
            # tmp[j] = pos2word[tmp[j]]       # convert index to word
        sort_table.append(tmp)

    return np.array(sort_table)


def text2matrix(s, sliding_window=3, target_width=5):
    """

    """
    (wlist, tokenwords, word2pos, pos2word) = count_words(s)
    word2id = {k: v for k, v in zip(wlist, range(len(wlist)))}
    wordslist_length = len(wlist)

    AM_table = [[0 for i in range(wordslist_length)] for j in range(wordslist_length)]

    # generate occurred matrix with sliding_window
    for num in range(len(tokenwords) - sliding_window + 1):
        for i in range(sliding_window - 1):
            for j in range(i + 1, sliding_window):
                AM_table[wlist.index(tokenwords[num + i])][wlist.index(tokenwords[num + j])] += 1
                AM_table[wlist.index(tokenwords[num + j])][wlist.index(tokenwords[num + i])] += 1

    related_tables = {}
    for i in range(wordslist_length):
        related_tables[i] = [[index, num] for index, num in enumerate(AM_table[i]) if num > 0 and index != i]
        related_tables[i].sort(key=lambda x: x[1], reverse=True)
        related_tables[i] = [element[0] for element in related_tables[i]]

    TD_table = [[-1 for i in range(target_width)] for j in range(wordslist_length)]
    for i in range(wordslist_length):
        fill_table(TD_table[i], related_tables, target_width, [i])  # fill TD table with -1

    # TD_table = reorder(TD_table, word2pos, pos2word, wlist, word2id)

    # convert id to words: arrange word matrix
    awm = []
    for row in TD_table:
        awm.append([pos2word[i] for i in row])
    return wlist, awm  # ,TD_table