from torch import nn
import torch
from config import device

class LSTM(nn.Module):
    """
    LSTM model with embedding layer
    """
    def __init__(self, n_vocab, padding_idx, embedding_weights=None, embedding_size=50, hidden_size=128,
                 num_layers=2) -> None:
        super(LSTM, self).__init__()
        self.n_vocab = n_vocab
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_size, padding_idx=padding_idx)

        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, n_vocab)

    def forward(self, x, seq_len, state):
        x = self.embedding(x)
        x = self.dropout(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, enforce_sorted=False)
        output, (h, c) = self.LSTM(packed, state)
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        x = self.fc(x)
        return x, (h, c)

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
