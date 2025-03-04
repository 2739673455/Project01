import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 文本处理
data = pd.read_csv("data/date.csv")
# 构建词表
source_vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + list(set("".join(data["source"])))
target_vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + list(set("".join(data["target"])))
# 词表大小
input_size = len(source_vocab)
output_size = len(target_vocab)
# 词到索引的映射
source_word2idx = {word: index for index, word in enumerate(source_vocab)}
target_word2idx = {word: index for index, word in enumerate(target_vocab)}
# 语料索引化，并添加起止符
source_idx = [
    [source_word2idx["<bos>"]]
    + [source_word2idx.get(word, source_word2idx["<unk>"]) for word in line]
    + [source_word2idx["<eos>"]]
    for line in data["source"]
]
target_idx = [
    [target_word2idx["<bos>"]]
    + [target_word2idx.get(word, target_word2idx["<unk>"]) for word in line]
    + [target_word2idx["<eos>"]]
    for line in data["target"]
]
# 计算最大的序列长度，并对长度不足的序列使用<pad>补齐
max_source_length = max([len(line) for line in source_idx])
max_target_length = max([len(line) for line in target_idx])
source_idx = [line + [source_word2idx["<pad>"]] * (max_source_length - len(line)) for line in source_idx]
target_idx = [line + [target_word2idx["<pad>"]] * (max_target_length - len(line)) for line in target_idx]


# 构建数据集
class Seq2SeqDataset(Dataset):
    def __init__(self, source_idx, target_idx):
        self.source = torch.LongTensor(source_idx)  # [num_samples, seq_len]
        self.target = torch.LongTensor(target_idx)  # [num_samples, seq_len]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


def collate_fn(batch):
    source, target = zip(*batch)
    source = torch.stack(source)  # [seq_len, batch_size]
    target = torch.stack(target)  # [seq_len, batch_size]
    return source, target


dataset = Seq2SeqDataset(source_idx, target_idx)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# 模型搭建
class Encoder(nn.Module):
    """编码器"""

    def __init__(self, input_size, hidden_size):
        # input_size:输入词表大小
        # hidden_size:隐藏层维度数量
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # 词嵌入层
        self.lstm = nn.LSTM(hidden_size, hidden_size)  # lstm层

    def forward(self, input):
        # input: [seq_len, batch_size]
        embedded = self.embedding(input)  # [seq_len, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm(embedded)
        # output: [seq_len, batch_size, hidden_size]
        # hidden: [1, batch_size, hidden_size]
        # cell: [1, batch_size, hidden_size]
        return output, hidden, cell


class Attention(nn.Module):
    """注意力机制"""

    def __init__(self, hidden_size):
        super().__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [seq_len, batch_size, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        encoder_outputs1 = encoder_outputs.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]
        decoder_hidden = decoder_hidden.permute(1, 0, 2)  # [batch_size, 1, hidden_size]
        score = self.W(torch.tanh(self.Wq(decoder_hidden) + self.Wk(encoder_outputs1)))  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(score, dim=1).transpose(1, 2)  # [batch_size, 1, seq_len]
        # bmm为批量矩阵乘法，对批量中的每个样本独立执行矩阵乘法。bmm([batch_size,n,m],[batch_size,m,p]) -> [batch_size,n,p]
        context = torch.bmm(attention_weights, encoder_outputs1)  # [batch_size, 1, hidden_size]
        return context.transpose(0, 1)  # [1, batch_size, hidden_size]


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)  # 词嵌入层
        self.attention = Attention(hidden_size)  # 注意力机制
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size)  # lstm层
        self.linear = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [1, batch_size]
        embedded = self.embedding(input)  # [1, batch_size, hidden_size]
        context = self.attention(encoder_outputs, hidden)  # [1, batch_size, hidden_size]
        input = torch.cat((embedded, context), dim=2)  # 拼接输入向量和上下文向量，# [1, batch_size, hidden_size*2]
        output, (hidden, cell) = self.lstm(input, (hidden, cell))  # [1, batch_size, hidden_size]
        output = self.linear(output)  # [1, batch_size, output_size]
        return output, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: [seq_len, batch_size]
        # target: [seq_len, batch_size]
        batch_size = source.size(1)  # 批量大小
        target_len = target.size(0)  # 目标序列长度
        target_vocab_size = self.decoder.linear.out_features  # 目标词表大小
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)  # 初始化输出张量
        encoder_outputs, hidden, cell = self.encoder(source)  # 获取编码器输出，隐藏层状态，细胞状态
        input = target[0, :]  # 目标序列第一个词<SOS>作为解码器输入
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input.unsqueeze(0), hidden, cell, encoder_outputs)  # 获取解码器输出
            outputs[t:,] = output.squeeze(1)  # 将解码器输出添加到输出中
            print(outputs)
            # 教师强制是指在解码器训练时，使用真实目标序值作为输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio  # 随机使用教师强制
            input = target[t] if teacher_force else output.argmax(2).squeeze(0)  # 根据是否使用教师强制选择输入词
        return outputs


# 模型训练
hidden_size = 64
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(output_size, hidden_size)
model = Seq2Seq(encoder, decoder)
criterion = nn.CrossEntropyLoss(ignore_index=target_word2idx["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for source, target in dataloader:
        print(source.shape)
        print(target.shape)
        output = model(source, target)
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
