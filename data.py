def data_read(pth, mode):
    pth += '/'
    # 读入训练语料还是预测语料
    if mode == 'train':
        en_txt_file_name = 'en.txt'
    else:
        en_txt_file_name = 'predict.txt'

    cn_sentences = []
    with open(pth + 'cn.txt', 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            cn_sentences.append(sentence)
    en_sentences = []
    with open(pth + en_txt_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            en_sentences.append(sentence)

    # 读入词典，从0开始编号
    cn_dic = {}
    num = 0
    with open(pth + 'cn.txt.vocab.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            word, _ = line.strip().split('\t')
            cn_dic[word] = int(num)
            num += 1
    en_dic = {}
    num = 0
    with open(pth + 'en.txt.vocab.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            word, _ = line.strip().split('\t')
            en_dic[word] = int(num)
            num += 1

    target_number_to_word = {v: k for k, v in cn_dic.items()}

    # 为decoder的输入和输出添加起始符号和结束符号
    cn_sentences_dec_input = ['<S> ' + sentence for sentence in cn_sentences]
    cn_sentences_dec_output = [sentence + ' </S>' for sentence in cn_sentences]

    data_train = [en_sentences, cn_sentences_dec_input, cn_sentences_dec_output]

    # 词典最大索引号
    input_dic_max_index = max(en_dic.values()) + 1
    target_dic_max_index = max(cn_dic.values()) + 1

    return input_dic_max_index, target_dic_max_index, en_dic, cn_dic, data_train, target_number_to_word