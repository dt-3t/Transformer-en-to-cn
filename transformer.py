import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from data import download_data, data_read
import os
import argparse


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

    enc_inputs_tensor , dec_inputs_tensor , target_tensor = torch.LongTensor(enc_inputs) , torch.LongTensor(dec_inputs) , torch.LongTensor(target)
    return enc_inputs_tensor, dec_inputs_tensor, target_tensor


# 在训练时，我们需要对输入进行mask。此函数用于生成mask矩阵，即一个上三角矩阵，其中对角线以下的元素全为0，对角线及以上的元素全为1。
def get_sequence_mask_sign(seq):
    # 输入的seq形状为(batch_size, tgt_len)，故mask_sign形状为(batch_size, tgt_len, tgt_len)
    mask_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 先创建全为1的矩阵，然后使用torch.triu()函数将数组的下三角部分设置为零。
    mask_sign = torch.triu(torch.ones(mask_shape), diagonal=1)
    return mask_sign.to(device)


# 获得pad的mask矩阵。对K中的pad标记，拓展到所有Q。
def get_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_sign = seq_k.eq(pad_num).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_sign.to(device)
    # pad_sign形状: (batch_size, len_q, len_k)


# 计算注意力分数的函数
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


# 其实包括了MultiHeadAttention，残差连接，和LayerNorm。
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # Q形状: (batch_size, len_q, d_model)
        # K形状: (batch_size, len_k, d_model)
        # V形状: (batch_size, len_v(=len_k), d_model)

        residual = Q  # 记录下输入进来的Q，后面将作为残差加入到输出中。
        batch_size, len_q, _ = Q.size()

        q_s = self.W_Q(Q)
        q_s = q_s.reshape(batch_size, len_q, n_heads, d_k)
        q_s = q_s.transpose(1, 2)
        k_s = self.W_K(K).reshape(batch_size, len_q, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).reshape(batch_size, len_q, n_heads, d_v).transpose(1, 2)
        # q_s形状: (batch_size, n_heads, len_q, d_k)
        # k_s形状: (batch_size, n_heads, len_k, d_k)
        # v_s形状: (batch_size, n_heads, len_k, d_v)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask形状: (batch_size,len_q, len_k) -> (batch_size, n_heads, len_q, len_k)

        context = scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)
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
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncodingLayer, self).__init__()
        # d_model: 词向量的维度，默认是512
        # max_len: 句子的最大长度(即单词数），默认是5000

        # 计算位置编码
        pos = torch.arange(0, max_len)  # 公式中的pos，0到max_len-1，形状为(max_len, )
        pos = pos.unsqueeze(1).repeat(1, d_model)  # 拓展形状为(max_len, d_model)
        tmp = torch.zeros(d_model)  # 声明pos要乘上的部分。现在还不是矩阵，只是一个向量，形状为(d_model, )
        tmp[0::2] = tmp[1::2] = torch.arange(0, d_model, 2)  # 填入2i的值
        tmp = torch.exp((-math.log(10000.0) * (tmp / d_model)))  # 计算pos要乘上的部分
        tmp = tmp.unsqueeze(0).repeat(max_len, 1)  # 拓展形状为(max_len, d_model)
        pe = torch.sin(pos * tmp)  # 形状为(max_len, d_model)。(max_len, d_model)矩阵，作为一个批次的位置编码
        pe[:, 0::2] = torch.sin(pe[:, 0::2])  # dim=2i，使用sin
        pe[:, 1::2] = torch.cos(pe[:, 1::2])  # dim=2i+1，使用cos
        pe = pe.unsqueeze(0)  # 拓展形状为(1, max_len, d_model)，方便后续在batch_size维度上拓展
        # pe.requires_grad = False  # 位置编码不更新
        self.register_buffer('pe', pe, False)
        # self.pe = pe

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x的形状: (batch_size, seq_len, d_model)
        pe = self.pe.repeat(x.size(0), 1, 1)[:, :x.size(1), :]  # 在batch_size维度上复制，并截取到seq_len长度
        x = x + pe
        return self.dropout(x)


# 多头注意力机制和前馈神经网络的整体，作为EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = FeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # 注意力层
        enc_outputs = self.pos_ffn(enc_outputs)  # 前馈网络层
        return enc_outputs


#  包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(input_dic_max_index, d_model)  # 用于将小于等于src_vocab_size-1的数字映射为d_model维向量
        self.pos_emb = PositionalEncodingLayer(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 堆叠n_layers层EncoderLayer

    def forward(self, enc_inputs):
        # enc_inputs形状: (batch_size, src_len)

        enc_outputs = self.src_emb(enc_inputs)  # 词嵌入。enc_outputs形状: (batch_size, src_len, d_model)
        enc_outputs = self.pos_emb(enc_outputs)  # 加上位置编码。enc_outputs形状不变

        pad_mask_sign = get_pad_mask(enc_inputs, enc_inputs)  # 得到对于pad的mask矩阵

        # 进入N层EncoderLayer
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, pad_mask_sign)
        return enc_outputs  # 返回整个Encoder的输出和每一层的注意力分数


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # 第一个注意力层
        self.dec_enc_dec_attn = MultiHeadAttention()  # 第二个注意力层
        self.pos_ffn = FeedForwardNet()  # 前馈网络层

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # 第一层为自注意力层，Q、K、V都是dec_inputs
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 第二层为交互注意力层，Q是解码器第一层输出，K、V是编码器的输出
        dec_outputs = self.dec_enc_dec_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # 前馈网络层
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(target_dic_max_index, d_model)  # 用于将小于等于tgt_vocab_size-1的数字映射为d_model维向量
        self.pos_emb = PositionalEncodingLayer(d_model)  # 位置编码层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # 堆叠n_layers层DecoderLayer

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]

        # 获得dec_inputs的pad的mask矩阵，形状(batch_size, tgt_len, tgt_len)
        dec_self_attn_pad_mask = get_pad_mask(dec_inputs, dec_inputs)
        # 获得dec_inputs的上三角mask矩阵
        dec_self_attn_subsequent_mask = get_sequence_mask_sign(dec_inputs)
        # 合并mask矩阵。有一个为1则为1，形状(batch_size, tgt_len, tgt_len)
        dec_self_mask_sign = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # dec_inputs对enc_inputs的pad的mask矩阵，形状(batch_size, tgt_len, src_len)
        dec_mid_mask_sign = get_pad_mask(dec_inputs, enc_inputs)

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect script with --cfg option.")
    parser.add_argument("--cfg", required=True, type=str, help="Train or predict.")
    return parser.parse_args()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数
    batch_size = 100
    d_model = 512  # 词嵌入维度
    d_ff = 2048  # 前馈网络隐藏维度
    d_k = d_v = 64  # 每个头的维度
    n_layers = 6  # EncoderLayer和DecoderLayer堆叠的数量
    n_heads = 8  # 多头注意力的头数
    max_len = 100  # 句子最大长度
    unk_num = 1  # 未知词的编号
    pad_num = 0  # pad符号的编号
    start_num = 2  # 句子起始符号
    end_num = 3  # 句子结束符号
    lr = 0.0002  # 学习率
    eph = 60  # 训练轮数
    model_pth = 'model.pth'  # 模型路径

    args = parse_arguments()
    mode = args.cfg

    # 如果不存在数据集，则下载数据集
    if not os.path.exists('data'):
        download_data()

    # 读入数据
    dataset_pth = 'data/'  # 数据集路径
    input_dic_max_index, target_dic_max_index, en_dic, cn_dic, data_train, target_number_to_word = data_read(dataset_pth, mode)

    # 模型
    model = Transformer()
    model.to(device)

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.85)

    # 将字符串转换为索引
    enc_inputs, dec_inputs, target = word_to_index(en_dic, cn_dic, data_train)
    enc_inputs, dec_inputs, target = enc_inputs.to(device), dec_inputs.to(device), target.to(device)

    if mode == 'predict':
        train_dataset = TensorDataset(enc_inputs)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        model.load_state_dict(torch.load(model_pth))
        model.eval()
        predict = []
        for batch_enc_inputs in train_loader:
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
            predict.append(' '.join(output_words))
        with open('predict.txt', 'w', encoding='utf-8') as file:
            for line in predict:
                file.write(line + '\n')
        exit()

    # 将数据放入DataLoader
    train_dataset = TensorDataset(enc_inputs, dec_inputs, target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练
    model.train()
    for epoch in range(eph):
        for batch_enc_inputs, batch_dec_inputs, batch_target in train_loader:
            optimizer.zero_grad()

            batch_outputs = model(batch_enc_inputs, batch_dec_inputs)

            # 每10个epoch打印一次结果
            if epoch % 10 == 0:
                batch_predict = batch_outputs.data.max(2, keepdim=True)[1]
                tmp = batch_predict[0].squeeze().cpu().numpy()
                output_words = [target_number_to_word[x] for i, x in enumerate(tmp) if x not in {0, 3}]
                print(' '.join(output_words))

            batch_loss = criterion(batch_outputs.transpose(-1, -2), batch_target)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(batch_loss), 'lr =',
                  '{:.6f}'.format(scheduler.get_last_lr()[0]))

            batch_loss.backward()
            optimizer.step()

            scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
