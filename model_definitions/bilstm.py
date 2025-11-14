import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """
    BiLSTM + Bahdanau-style Attention with embedding layers.

    Treats pain surveys (0-4) as categorical, not continuous.
    Each pain_survey column gets its own embedding layer.
    Embeddings are concatenated with continuous features before BiLSTM.
    """
    def __init__(
        self,
        num_continuous_features,
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=5,
        embedding_dim=3,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.3,
        **kwargs
    ):
        super().__init__()

        self.num_continuous_features = num_continuous_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_pain_surveys = num_pain_surveys
        self.embedding_dim = embedding_dim

        # Embedding layers for pain surveys
        self.pain_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_pain_levels, embedding_dim=embedding_dim)
            for _ in range(num_pain_surveys)
        ])
        
        # Total input features after embedding
        total_input_features = num_continuous_features + (num_pain_surveys * embedding_dim)

        # -----------------------
        # 1) BiLSTM stack
        # -----------------------
        self.lstm = nn.LSTM(
            input_size=total_input_features,
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

    def forward(self, categorical_input, continuous_input):
        """
        Args:
            categorical_input: (batch_size, seq_length, num_pain_surveys)
            continuous_input: (batch_size, seq_length, num_continuous_features)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed each pain survey column separately
        embedded_pain_surveys = []
        for i in range(self.num_pain_surveys):
            pain_col = categorical_input[:, :, i].long()
            embedded = self.pain_embeddings[i](pain_col)
            embedded_pain_surveys.append(embedded)
        
        # Concatenate embedded pain surveys
        embedded_pain = torch.cat(embedded_pain_surveys, dim=2)
        
        # Concatenate with continuous features
        x = torch.cat([continuous_input, embedded_pain], dim=2)
        
        # 1) BiLSTM
        # lstm_out: (B, T, 2*H)
        lstm_out, _ = self.lstm(x)

        # 2) Attention pooling sui time-step
        context, attn_weights = self.attention(lstm_out)  # (B, 2H), (B, T)

        # 3) Classificazione
        context = self.dropout(context)
        logits = self.classifier(context)  # (B, num_classes)

        return logits
