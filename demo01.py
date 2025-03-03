import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. 定义字符集
INPUT_CHARS = "0123456789/"  # 输入字符集
OUTPUT_CHARS = "0123456789-"  # 输出字符集
SOS_token = 0  # 开始标记
EOS_token = 1  # 结束标记


# 2. 创建词汇表
class Vocab:
    def __init__(self, chars):
        self.char2index = {"SOS": 0, "EOS": 1}
        self.index2char = {0: "SOS", 1: "EOS"}
        for i, char in enumerate(chars, 2):
            self.char2index[char] = i
            self.index2char[i] = char
        self.n_chars = len(self.char2index)


input_vocab = Vocab(INPUT_CHARS)
output_vocab = Vocab(OUTPUT_CHARS)


# 3. 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)  # 去掉batch_first=True

    def forward(self, x):
        # x: [seq_len, batch_size]
        embedded = self.embedding(x)  # [seq_len, batch_size, hidden_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


# 4. Bahdanau注意力机制
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]

        decoder_hidden = decoder_hidden.transpose(0, 1)  # [batch_size, 1, hidden_size]
        # 调整维度以匹配
        energy = torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_outputs.transpose(0, 1)))
        attention_scores = self.Va(energy.transpose(0, 1)).squeeze(2)  # [seq_len, batch_size]
        attention_weights = torch.softmax(attention_scores, dim=0).unsqueeze(0)  # [1, seq_len, batch_size]
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs.transpose(0, 1))
        return context.transpose(0, 1), attention_weights  # [batch_size, 1, hidden_size]


# 5. 解码器
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size)  # 去掉batch_first=True
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [1, batch_size]
        embedded = self.embedding(x)  # [1, batch_size, hidden_size]
        context, attention_weights = self.attention(hidden, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, hidden_size*2]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.out(output.squeeze(0))  # [batch_size, output_size]
        return prediction, hidden, cell, attention_weights


# 6. Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: [seq_len, batch_size]
        # target: [seq_len, batch_size]
        batch_size = source.size(1)
        target_len = target.size(0)
        outputs = torch.zeros(target_len, batch_size, output_vocab.n_chars)

        encoder_outputs, hidden, cell = self.encoder(source)

        # 第一个输入是SOS
        decoder_input = torch.tensor([SOS_token] * batch_size).unsqueeze(0)

        for t in range(target_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[t] = output

            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[t].unsqueeze(0) if teacher_force else top1.unsqueeze(0)

        return outputs


# 7. 数据准备
def date_to_tensor(date_str, vocab, max_length=10):
    tensor = torch.zeros(max_length, dtype=torch.long)
    for i, char in enumerate(date_str):
        tensor[i] = vocab.char2index[char]
    tensor[len(date_str) - 1] = EOS_token
    return tensor


# 8. 训练函数
def train(model, optimizer, criterion, n_epochs=1000):
    for epoch in range(n_epochs):
        # 示例数据
        input_date = "03/03/2025"
        target_date = "2025-03-03"

        input_tensor = date_to_tensor(input_date, input_vocab).unsqueeze(1)  # [seq_len, 1]
        target_tensor = date_to_tensor(target_date, output_vocab).unsqueeze(1)  # [seq_len, 1]

        optimizer.zero_grad()
        output = model(input_tensor, target_tensor)

        loss = criterion(output.view(-1, output_vocab.n_chars), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 9. 主程序
hidden_size = 256
encoder = Encoder(input_vocab.n_chars, hidden_size)
decoder = Decoder(output_vocab.n_chars, hidden_size)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=EOS_token)

# 开始训练
train(model, optimizer, criterion)


# 10. 预测函数
def predict(model, date_str):
    model.eval()
    with torch.no_grad():
        input_tensor = date_to_tensor(date_str, input_vocab).unsqueeze(1)  # [seq_len, 1]
        encoder_outputs, hidden, cell = model.encoder(input_tensor)

        decoder_input = torch.tensor([SOS_token]).unsqueeze(0)  # [1, 1]
        output_seq = []

        for _ in range(10):  # max length
            output, hidden, cell, _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            pred = output.argmax(1).item()
            if pred == EOS_token:
                break
            output_seq.append(output_vocab.index2char[pred])
            decoder_input = torch.tensor([pred]).unsqueeze(0)

        return "".join(output_seq)


# 测试
print(predict(model, "03/03/2025"))  # 应该输出 "2025-03-03"
