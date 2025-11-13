import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """
    BiLSTM + Bahdanau-style Attention per time-series classification.

    Input:  x  -> (batch_size, seq_len, input_size)
    Output: logits -> (batch_size, num_classes)
    """
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.3,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # -----------------------
        # 1) BiLSTM stack
        # -----------------------
        # batch_first=True → (B, T, F)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Dimensione dell'output LSTM: hidden_size * 2 (fw + bw)
        lstm_out_dim = hidden_size * 2

        # -----------------------
        # 2) Bahdanau-style Attention
        # -----------------------
        # score_t = v^T * tanh(W_h * h_t)
        self.attn_W = nn.Linear(lstm_out_dim, lstm_out_dim, bias=True)
        self.attn_v = nn.Linear(lstm_out_dim, 1, bias=False)

        # -----------------------
        # 3) Classification head
        # -----------------------
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

    def attention(self, lstm_out):
        """
        lstm_out: (B, T, H*2)
        ritorna:
            context: (B, H*2)
            attn_weights: (B, T)
        """
        # u_t = tanh(W_h * h_t)
        # lstm_out: (B, T, D)
        u = torch.tanh(self.attn_W(lstm_out))      # (B, T, D)

        # score_t = v^T u_t
        scores = self.attn_v(u).squeeze(-1)       # (B, T)

        # α_t = softmax(scores)
        attn_weights = F.softmax(scores, dim=-1)  # (B, T)

        # context = Σ_t α_t * h_t
        # (B, 1, T) @ (B, T, D) -> (B, 1, D) -> (B, D)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        # 1) BiLSTM
        # lstm_out: (B, T, 2*H)
        lstm_out, _ = self.lstm(x)

        # 2) Attention pooling sui time-step
        context, attn_weights = self.attention(lstm_out)  # (B, 2H), (B, T)

        # 3) Classificazione
        context = self.dropout(context)
        logits = self.classifier(context)  # (B, num_classes)

        return logits
