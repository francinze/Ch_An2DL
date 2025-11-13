import torch.nn as nn
import torch

class CNN1DClassifier_bidirectional(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size,
        dropout=0.4,
        bidirectional=False,
        rnn_type='GRU',     # 'RNN', 'LSTM', 'GRU'
        num_layers=1
    ):
        super().__init__()
        
        # ====== CNN BLOCK ======
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # ====== RNN BLOCK ======
        self.rnn_type     = rnn_type
        self.num_layers   = num_layers
        self.hidden_size  = hidden_size
        self.bidirectional = bidirectional

        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")
        
        rnn_module = rnn_map[rnn_type]

        # Dropout solo tra i layer dell'RNN (se num_layers > 1)
        rnn_dropout = dropout if num_layers > 1 else 0.0

        # L'RNN prende in input i canali finali della CNN: 256
        self.rnn = rnn_module(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )

        # ====== CLASSIFIER ======
        if self.bidirectional:
            classifier_input_size = hidden_size * 2  # concat fwd + bwd
        else:
            classifier_input_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        # CNN vuole (batch, channels, seq_len)
        x = x.transpose(1, 2)   # -> (batch, features, seq_len)

        # ----- CNN -----
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        # x: (batch, 256, seq_len_cnn)

        # ----- RNN -----
        # RNN vuole (batch, seq_len, features)
        x_rnn_in = x.transpose(1, 2)  # -> (batch, seq_len_cnn, 256)

        rnn_out, hidden = self.rnn(x_rnn_in)
        # rnn_out: (batch, seq_len_cnn, hidden_size * num_directions)
        # hidden:  (num_layers * num_directions, batch, hidden_size)
        # per LSTM: hidden = (h_n, c_n), ci serve solo h_n
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]

        if self.bidirectional:
            # (num_layers, 2, batch, hidden_size)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            # ultimi stati fwd e bwd dell’ultima layer
            hidden_to_classify = torch.cat(
                [hidden[-1, 0, :, :], hidden[-1, 1, :, :]],
                dim=1
            )  # -> (batch, hidden_size * 2)
        else:
            hidden_to_classify = hidden[-1]  # (batch, hidden_size)

        logits = self.fc(hidden_to_classify)
        return logits