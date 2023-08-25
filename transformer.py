import argparse
import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split

from data import data_read


# 将句子中的单词转换为词典中的索引.
def word_to_index(dic_input, dic_target, data):
    enc_inputs, dec_inputs, target = [], [], []
    for seq in data[0]:
        enc_inputs.append(
            [dic_input.get(word, unk_num) for word in seq.split()] + [pad_num] * (max_len - len(seq.split())))
    for seq in data[1]:
        dec_inputs.append(
            [dic_target.get(word, unk_num) for word in seq.split()] + [pad_num] * (max_len - len(seq.split())))
    for seq in data[2]:
        target.append(
            [dic_target.get(word, unk_num) for word in seq.split()] + [pad_num] * (max_len - len(seq.split())))

    enc_inputs_tensor, dec_inputs_tensor, target_tensor = torch.LongTensor(enc_inputs), torch.LongTensor(
        dec_inputs), torch.LongTensor(target)
    return enc_inputs_tensor, dec_inputs_tensor, target_tensor


# 在训练时，我们需要对输入进行mask。此函数用于生成mask矩阵，即一个上三角矩阵，其中对角线以下的元素全为0，对角线及以上的元素全为1。
def get_sequence_mask_sign(seq):
    # 先创建全为1的矩阵，然后使用torch.triu()函数将数组的下三角部分设置为零。
    mask_sign = torch.triu(torch.ones(seq.size(0), max_len, max_len), diagonal=1)
    return mask_sign.to(device)


# 获得pad的mask矩阵。输入的是K的seqence
def get_pad_mask(seq):
    pad_sign = seq.eq(pad_num)
    pad_sign = pad_sign.unsqueeze(1)
    pad_sign = pad_sign.repeat(1, max_len, 1)
    return pad_sign.to(device)
    # pad_sign形状: (batch_size, max_len, max_len)


# 带掩码的点积注意力机制
def scaled_dot_product_attention(Q, K, V, mask_sign):
    # Q形状: (batch_size, n_heads, len_q, d_k)
    # K形状: (batch_size, n_heads, len_k, d_k)
    # V形状: (batch_size, n_heads, len_v(=len_k), d_v)
    # batch_size代表多个句子，n_heads代表多个注意力头，len_x代表句子长度（即单词个数），d_x代表每个单词的词向量维度。
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # 形状(batch_size, n_heads, len_q, len_k), 即在d_k维度上相乘相加。
    # 其中K.transpose(-1, -2)是对K的最后两个维度进行转置, (batch_size, n_heads,len_k, d_k) -> (batch_size, n_heads, d_k, len_k)
    scores.masked_fill_(mask_sign, -1e9)  # 将attn_mask中为True的位置置为-1e9，即负无穷，这样softmax之后就几乎为0了
    scores = nn.Softmax(dim=-1)(scores)  # 在最后一个维度（即len_k维度）进行softmax
    context = torch.matmul(scores, V)  # 将注意力分数矩阵与V相乘，得到输出。形状(batch_size, n_heads, len_q, d_v)
    return context  # 返回输出和注意力权重矩阵


# 包括了MultiHeadAttention，残差连接，和LayerNorm。
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask_sign):
        # Q形状: (batch_size, len_q, d_model)
        # K形状: (batch_size, len_k, d_model)
        # V形状: (batch_size, len_v(=len_k), d_model)

        residual = Q  # 记录下输入进来的Q，后面将作为残差加入到输出中。
        batch_size, len_q, _ = Q.size()

        q_s = self.W_Q(Q)  # q_s形状: (batch_size, len_q, n_heads * d_k)
        q_s = q_s.reshape(batch_size, len_q, n_heads, d_k)
        q_s = q_s.transpose(1, 2)
        k_s = self.W_K(K).reshape(batch_size, len_q, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).reshape(batch_size, len_q, n_heads, d_v).transpose(1, 2)
        # q_s形状: (batch_size, n_heads, len_q, d_k)
        # k_s形状: (batch_size, n_heads, len_k, d_k)
        # v_s形状: (batch_size, n_heads, len_k, d_v)

        mask_sign = mask_sign.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask形状: (batch_size,len_q, len_k) -> (batch_size, n_heads, len_q, len_k)

        context = scaled_dot_product_attention(q_s, k_s, v_s, mask_sign)
        context = context.transpose(1, 2).reshape(batch_size, len_q, -1)
        # context形状: (batch_size, n_heads, len_q, d_v) -> (batch_size, len_q, n_heads * d_v)
        output = self.linear(context)  # output形状: (batch_size, len_q, d_model)
        output = output + residual  # 残差连接
        output = self.layer_norm(output)  # LayerNorm层
        return output
        # output形状: (batch_size, len_q, d_model)


# 事实上包含了FeedForward，残差连接，和LayerNorm。
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, inputs):
        # inputs形状: (batch_size, seq_len, d_model)
        residual = inputs
        output = self.net(inputs)
        output += residual
        return output
        # output形状和inputs相同: (batch_size, seq_len, d_model)


# 将x加上位置编码
class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len_tmp=5000):
        super(PositionalEncodingLayer, self).__init__()
        # d_model: 词向量的维度，默认是512
        # max_len: 句子的最大长度(即单词数），默认是5000

        # 计算位置编码
        pos = torch.arange(0, max_len_tmp)  # 公式中的pos，0到max_len-1，形状为(max_len, )
        pos = pos.unsqueeze(1).repeat(1, d_model)  # 拓展形状为(max_len, d_model)
        tmp = torch.zeros(d_model)  # 声明pos要乘上的部分。现在还不是矩阵，只是一个向量，形状为(d_model, )
        tmp[0::2] = tmp[1::2] = torch.arange(0, d_model, 2)  # 填入2i的值
        tmp = torch.exp((-math.log(10000.0) * (tmp / d_model)))  # 计算pos要乘上的部分
        tmp = tmp.unsqueeze(0).repeat(max_len_tmp, 1)  # 拓展形状为(max_len, d_model)
        pe = torch.sin(pos * tmp)  # 形状为(max_len, d_model)。(max_len, d_model)矩阵，作为一个批次的位置编码
        pe[:, 0::2] = torch.sin(pe[:, 0::2])  # dim=2i，使用sin
        pe[:, 1::2] = torch.cos(pe[:, 1::2])  # dim=2i+1，使用cos
        pe = pe.unsqueeze(0).repeat(1500, 1, 1)  # 拓展形状为(1500, max_len, d_model)
        self.register_buffer('pe', pe, False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x的形状: (batch_size, seq_len, d_model)
        pe = self.pe[:x.size(0), :max_len, :]  # 在batch_size维度上复制，并截取到seq_len长度
        x = x + pe
        return self.dropout(x)


# 多头注意力机制和前馈神经网络的整体，作为EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention()
        self.ffn_net = FeedForwardNet()

    def forward(self, enc_inputs, mask_sign):
        enc_outputs = self.self_attention(enc_inputs, enc_inputs, enc_inputs, mask_sign)  # 注意力层
        enc_outputs = self.ffn_net(enc_outputs)  # 前馈网络层
        return enc_outputs


#  包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dic_max_index, d_model)  # 用于将小于等于src_vocab_size-1的数字映射为d_model维向量
        self.position = PositionalEncodingLayer(d_model, max_len_tmp=max_len)  # 位置编码层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 堆叠n_layers层EncoderLayer

    def forward(self, enc_inputs):
        # enc_inputs形状: (batch_size, src_len)

        enc_outputs = self.embedding(enc_inputs)  # 词嵌入。enc_outputs形状: (batch_size, src_len, d_model)
        enc_outputs = self.position(enc_outputs)  # 加上位置编码。enc_outputs形状不变

        pad_mask_sign = get_pad_mask(enc_inputs)  # 得到对于pad的mask矩阵

        # 进入N层EncoderLayer
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, pad_mask_sign)
        return enc_outputs  # 返回整个Encoder的输出和每一层的注意力分数


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # 第一个注意力层
        self.dec_enc_dec_attn = MultiHeadAttention()  # 第二个注意力层
        self.ffn_net = FeedForwardNet()  # 前馈网络层

    def forward(self, dec_inputs, enc_outputs, mask_sign_1, mask_sign_2):
        # 第一层为自注意力层，Q、K、V都是dec_inputs
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, mask_sign_1)
        # 第二层为交互注意力层，Q是解码器第一层输出，K、V是编码器的输出
        dec_outputs = self.dec_enc_dec_attn(dec_outputs, enc_outputs, enc_outputs, mask_sign_2)
        dec_outputs = self.ffn_net(dec_outputs)  # 前馈网络层
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(target_dic_max_index, d_model)  # 用于将小于等于tgt_vocab_size-1的数字映射为d_model维向量
        self.pos_emb = PositionalEncodingLayer(d_model, max_len_tmp=max_len)  # 位置编码层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # 堆叠n_layers层DecoderLayer

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs和enc_inputs形状: (batch_size, max_len)
        # enc_outputs形状: (batch_size, max_len, d_model)
        dec_outputs = self.tgt_emb(dec_inputs)  # 形状(batch_size, max_len, d_model)
        dec_outputs = self.pos_emb(dec_outputs)

        # 获得dec_inputs的pad的mask矩阵，形状(batch_size, tgt_len, tgt_len)
        dec_self_attn_pad_mask = get_pad_mask(dec_inputs)
        # 获得dec_inputs的上三角mask矩阵
        dec_self_attn_subsequent_mask = get_sequence_mask_sign(dec_inputs)
        # 合并mask矩阵。有一个为1则为1，形状(batch_size, tgt_len, tgt_len)
        dec_self_mask_sign = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # dec_inputs对enc_inputs的pad的mask矩阵，形状(batch_size, tgt_len, src_len)
        dec_mid_mask_sign = get_pad_mask(enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_mask_sign, dec_mid_mask_sign)
        return dec_outputs


# 整体transformer网路分为编码层，解码层，输出层三部分
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # 编码部分
        self.decoder = Decoder()  # 解码部分
        self.to_dic = nn.Linear(d_model, target_dic_max_index, bias=False)  # 将d_model维映射为tgt_vocab_size维（即词表大小）

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)  # 将输入编码
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)  # 通过编码器输出和解码器输入，解码得到输出
        dec_logits = self.to_dic(dec_outputs)  # 将输出映射到词表大小，dec_logits形状为(batch_size, tgt_len, tgt_vocab_size)
        return dec_logits


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect script with --cfg option.")
    parser.add_argument("--cfg", required=True, type=str, help="Train or predict.")
    return parser.parse_args()


# 将句子中的单词转换为词典中的索引.
def fill_with_pad(tensor):
    mask = tensor.eq(end_num)
    nonzero_indices = torch.nonzero(mask)
    flg = torch.tensor([0] * nonzero_indices.shape[0])
    for pos_index in nonzero_indices:
        if flg[pos_index[0]] == 1:
            continue
        tensor[pos_index[0], pos_index[1] + 1:] = pad_num
        flg[pos_index[0]] = 1
    return tensor


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha  # 意义为学习率缩放倍数

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数
    batch_size = 550  # 批次大小
    d_model = 512  # 词嵌入维度
    d_ff = 2048  # 前馈网络隐藏维度
    d_k = d_v = 64  # 每个头的维度
    n_layers = 6  # EncoderLayer和DecoderLayer堆叠的数量
    n_heads = 8  # 多头注意力的头数
    max_len = 50  # 句子最大长度
    unk_num = 1  # 未知词的编号
    pad_num = 0  # pad符号的编号
    start_num = 2  # 句子起始符号
    end_num = 3  # 句子结束符号
    model_pth = 'model/train_8/last.pth'  # 模型路径
    predict_folder = 'predict'  # 预测文件夹
    dataset_pth = 'data'  # 数据集路径
    train_model_pth = 'model'  # 训练模型保存路径

    args = parse_arguments()
    mode = args.cfg

    # 读入数据

    input_dic_max_index, target_dic_max_index, en_dic, cn_dic, data_train, target_number_to_word = data_read(
        dataset_pth, mode)

    # 模型
    model = Transformer()
    model.to(device)

    # 将字符串转换为索引
    enc_inputs, dec_inputs, target = word_to_index(en_dic, cn_dic, data_train)
    enc_inputs, dec_inputs, target = enc_inputs.to(device), dec_inputs.to(device), target.to(device)

    if mode == 'predict':
        if not os.path.exists(predict_folder):
            os.mkdir(predict_folder)
        predict_dataset = TensorDataset(enc_inputs)
        predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load(model_pth))
        model.eval()
        predict = []
        sentence_index = 0
        for batch_enc_inputs in predict_loader:
            # 构造batch_dec_inputs，形状为(batch_size, max_len)，全部填充pad_num
            batch_enc_inputs = batch_enc_inputs[0]
            batch_dec_inputs = torch.full((1, max_len), pad_num, dtype=torch.long).to(device)
            batch_dec_inputs[:, 0] = start_num  # 将batch_dec_inputs的第一个位置设置为start_num
            # 开始预测
            now_pos = 0
            while now_pos < max_len - 1:
                batch_outputs = model(batch_enc_inputs, batch_dec_inputs)
                batch_predict = batch_outputs.data.max(2, keepdim=True)[1]
                tmp = batch_predict[0].squeeze().cpu().numpy()
                if tmp[now_pos] == end_num:  # 预测到结束符号则停止
                    break
                batch_dec_inputs[:, now_pos + 1] = tmp[now_pos]  # 将batch_dec_inputs的第now_pos+1个位置设置为预测值
                now_pos += 1
            # 将batch_dec_inputs中的数字转换为单词
            tmp = batch_dec_inputs[0].squeeze().cpu().numpy()
            output_words = [target_number_to_word[x] for i, x in enumerate(tmp) if x not in {0, 3}]
            predict.append(data_train[0][sentence_index])
            predict.append(' '.join(output_words))
            predict.append('')
            sentence_index += 1
        # 将model_pth中的'/'符号替换为'.'
        model_pth_tmp = model_pth.replace('/', '.')
        predict_file_name = os.path.join(model_pth_tmp[:-4] + '_predict')
        file_names = os.listdir(predict_folder)
        file_index_tmp = 0
        for file_name in file_names:
            if file_name[:-6] == predict_file_name:
                file_index_tmp += 1
        predict_file_name += '_' + str(file_index_tmp) + '.txt'
        file_pth = os.path.join(predict_folder, predict_file_name)
        with open(file_pth, 'w', encoding='utf-8') as file:
            for line in predict:
                file.write(line + '\n')
        exit()

    use_pretrain = False
    if use_pretrain:
        model.load_state_dict(torch.load(model_pth))
    # 将数据放入DataLoader
    train_dataset = TensorDataset(enc_inputs, dec_inputs, target)

    # 计算数据集中每个部分的数量
    total_samples = len(train_dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    # 根据划分比例随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if not os.path.exists(train_model_pth):
        os.mkdir(train_model_pth)
    folder_names = os.listdir(train_model_pth)
    folder_index = folder_names.__len__()
    train_model_pth = os.path.join(train_model_pth, 'train_' + str(folder_index))
    os.mkdir(train_model_pth)
    # 优化器
    eph = 90
    lr = 0.0002
    val_epoch = 75
    criterion = nn.CrossEntropyLoss(ignore_index=pad_num)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    warmup_epochs = 10
    keep_epochs = 0
    warmup_iters = len(train_loader) * warmup_epochs  # 学习率线性warmup
    warmup_factor = 0.0001
    decay_epoch = 20
    decay_gamma = 0.5
    decay_step_size = len(train_loader) * decay_epoch
    scheduler_1 = warmup_lr_scheduler(optimizer, warmup_iters=warmup_iters, warmup_factor=warmup_factor)
    scheduler_2 = lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_gamma)
    # 训练
    best_val_loss = 1e9
    print('Start training...')
    for epoch in range(eph):
        model.train()
        train_loss = 0
        if epoch <= warmup_epochs:
            scheduler = scheduler_1
        elif epoch > warmup_epochs + keep_epochs:
            scheduler = scheduler_2
        else:
            scheduler = None
        for batch_enc_inputs, batch_dec_inputs, batch_target in train_loader:
            optimizer.zero_grad()
            batch_outputs = model(batch_enc_inputs, batch_dec_inputs)
            batch_outputs = batch_outputs.view(-1, batch_outputs.size(-1))
            batch_target = batch_target.view(-1)
            batch_loss = criterion(batch_outputs, batch_target)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print('Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(train_loss), 'lr =',
              '{:.6f}'.format(current_lr))
        # 在验证集上测试
        if epoch >= val_epoch:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_enc_inputs, _, batch_target in val_loader:
                    # 构造batch_dec_inputs，形状为(batch_size, max_len)，全部填充pad_num
                    batch_dec_inputs = torch.full((batch_enc_inputs.size(0), max_len), pad_num, dtype=torch.long).to(
                        device)
                    batch_dec_inputs[:, 0] = start_num  # 将batch_dec_inputs的第一个位置设置为start_num
                    outputs_scores = torch.zeros(batch_enc_inputs.size(0), max_len, target_dic_max_index).to(device)
                    # 开始预测
                    now_pos = 0
                    while now_pos < max_len - 1:
                        batch_outputs = model(batch_enc_inputs, batch_dec_inputs)
                        outputs_tmp = batch_outputs[:, now_pos, :].squeeze()
                        outputs_scores[:, now_pos, :] = outputs_tmp
                        batch_predict = outputs_tmp.data.max(1, keepdim=True)[1]
                        batch_dec_inputs[:, now_pos + 1] = batch_predict.squeeze()
                        now_pos += 1
                    # 找到batch_dec_inputs中的end_num，将其后面的部分全部置为pad_num
                    # batch_dec_inputs = fill_with_pad(batch_dec_inputs)
                    outputs_scores = outputs_scores.view(-1, batch_outputs.size(-1))
                    batch_target = batch_target.view(-1)
                    val_loss = criterion(outputs_scores, batch_target)
                    # val_loss /= batch_enc_inputs.size(0)
                    # 将batch_dec_inputs[0]中的数字转换为单词，输出
                    tmp = batch_dec_inputs[0].squeeze().cpu().numpy()
                    output_words = [target_number_to_word[x] for i, x in enumerate(tmp) if x not in {0, 3}]
                    print(' '.join(output_words))
                current_lr = optimizer.param_groups[0]['lr']
                print('Epoch:', '%04d' % (epoch + 1), 'val_loss =', '{:.6f}'.format(val_loss), 'lr =',
                      '{:.6f}'.format(current_lr))
                print()
                # 保存最好的模型
                if val_loss < best_val_loss:
                    if os.path.exists(os.path.join(train_model_pth, 'best.pth')):
                        os.remove(os.path.join(train_model_pth, 'best.pth'))
                    torch.save(model.state_dict(), os.path.join(train_model_pth, 'best.pth'))
                    best_val_loss = val_loss

    # 保存最后的模型
    torch.save(model.state_dict(), os.path.join(train_model_pth, 'last.pth'))
