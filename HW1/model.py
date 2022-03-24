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

        out = torch.cat((avg_pool, max_pool), 1)

        out = self.relu(self.linear(out))
        out = self.dropout(out)
        out = self.out(out)

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
        self.lstm_ln = nn.LayerNorm(hidden_size * 2)

        self.softmax = nn.Softmax(dim=1)
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
        rnn_out = self.lstm_ln(rnn_out)

        out = self.softmax(rnn_out)
        out = self.dropout(out)
        out = self.out(out)

        return out
