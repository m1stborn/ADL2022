from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        # TODO: model architecture
        self.lstm = nn.LSTM(300, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    # def forward(self, batch) -> Dict[str, torch.Tensor]:
    def forward(self, batch):

        h_embedding = self.embed(batch)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)

        # print(h_lstm.size())
        # 8, 128, 128
        # print(avg_pool.size())
        # 8, 128
        # print(max_pool.size())
        # 8, 128

        out = torch.cat((avg_pool, max_pool), 1)
        # print(out.size())
        # 8, 256
        out = self.relu(self.linear(out))
        out = self.dropout(out)
        out = self.out(out)

        # print(out.size())
        # 8, 150
        return out


class SlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SlotClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.emb_ln = nn.LayerNorm(300)

        self.rnn = nn.LSTM(300, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        # self.rnn = nn.LSTM(300, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.lstm_ln = nn.LayerNorm(hidden_size * 2)

        # self.linear = nn.Linear(hidden_size * 2, 64)
        # self.linear = nn.Linear(hidden_size * 2, num_class)
        self.relu = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    # def forward(self, batch) -> Dict[str, torch.Tensor]:
    def forward(self, batch):

        h_embedding = self.embed(batch)
        h_embedding = self.emb_ln(h_embedding)

        rnn_out, _ = self.rnn(h_embedding, None)
        # batch_size x L x hidden_size*2
        rnn_out = self.lstm_ln(rnn_out)

        # out = self.linear(rnn_out)
        # batch_size x L x 64
        out = self.relu(rnn_out)
        out = self.dropout(out)
        out = self.out(out)

        return out


if __name__ == '__main__':
    rnn = nn.LSTM(300, 512, 2, bidirectional=True, batch_first=True)
    linear = nn.Linear(1024, 64)
    relu = nn.ReLU()
    dropout = nn.Dropout()
    linear2 = nn.Linear(64, 9)

    inputs = torch.randn(3, 10, 300)
    outputs, outputs2 = rnn(inputs, None)
    # outputs = linear(outputs)
    # outputs = relu(outputs)
    # outputs = dropout(outputs)
    # outputs = linear2(outputs)
    print(outputs.size(), outputs2[0].size())
    # loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    #
    # target = torch.empty((3, 10), dtype=torch.long).random_(9)
    # print(target.size())
    # loss = loss_function_1(outputs.transpose(1, 2), target)
    # print(loss.item())
    #
    # _, predicted = torch.max(outputs.data, 2)
    # print(predicted.size())
    # print(predicted)
