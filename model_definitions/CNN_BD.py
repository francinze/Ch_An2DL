import torch.nn as nn
import torch

class CNN1DClassifier_bidirectional(nn.Module):
    """
    CNN + RNN hybrid with embedding layers for categorical pain survey features.
    
    Treats pain surveys (0-4) as categorical, not continuous.
    Each pain_survey column gets its own embedding layer.
    Embeddings are concatenated with continuous features before CNN.
    """
    def __init__(
        self,
        num_continuous_features,
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=5,
        embedding_dim=3,
        hidden_size=128,
        dropout=0.4,
        bidirectional=False,
        rnn_type='GRU',
        num_layers=1
    ):
        super().__init__()
        
        self.num_pain_surveys = num_pain_surveys
        self.embedding_dim = embedding_dim
        
        # Embedding layers for pain surveys
        self.pain_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_pain_levels, embedding_dim=embedding_dim)
            for _ in range(num_pain_surveys)
        ])
        
        # Total input features after embedding
        total_input_features = num_continuous_features + (num_pain_surveys * embedding_dim)
        
        # ====== CNN BLOCK ======
        self.conv1 = nn.Conv1d(total_input_features, 64, kernel_size=5, padding=2)
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
        
        # Attention mechanism (optional)
        self.use_attention = True
        if self.use_attention:
            self.attn_W = nn.Linear(classifier_input_size, classifier_input_size, bias=True)
            self.attn_v = nn.Linear(classifier_input_size, 1, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def attention(self, rnn_out):
        """
        Bahdanau-style attention mechanism.
        
        Args:
            rnn_out: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: (batch_size, hidden_dim)
            attn_weights: (batch_size, seq_len)
        """
        # u_t = tanh(W_h * h_t)
        u = torch.tanh(self.attn_W(rnn_out))  # (B, T, D)
        
        # score_t = v^T u_t
        scores = self.attn_v(u).squeeze(-1)  # (B, T)
        
        # α_t = softmax(scores)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (B, T)
        
        # context = Σ_t α_t * h_t
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)
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
        
        # Use attention pooling or last hidden state
        if self.use_attention:
            # Attention-based pooling over all timesteps
            context, attn_weights = self.attention(rnn_out)  # (B, H)
        else:
            # hidden:  (num_layers * num_directions, batch, hidden_size)
            # per LSTM: hidden = (h_n, c_n), ci serve solo h_n
            if self.rnn_type == 'LSTM':
                hidden = hidden[0]

            if self.bidirectional:
                # (num_layers, 2, batch, hidden_size)
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
                # ultimi stati fwd e bwd dell'ultima layer
                context = torch.cat(
                    [hidden[-1, 0, :, :], hidden[-1, 1, :, :]],
                    dim=1
                )  # -> (batch, hidden_size * 2)
            else:
                context = hidden[-1]  # (batch, hidden_size)

        logits = self.fc(context)
        return logits