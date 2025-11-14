import torch
import torch.nn as nn


class RecurrentClassifier(nn.Module):
    """
    Generic RNN classifier (RNN, LSTM, GRU) with embedding layers.
    
    Treats pain surveys (0-4) as categorical, not continuous.
    Each pain_survey column gets its own embedding layer.
    Embeddings are concatenated with continuous features before RNN.
    """
    def __init__(
            self,
            num_continuous_features,
            hidden_size,
            num_layers,
            num_classes=3,
            num_pain_surveys=4,
            num_pain_levels=5,
            embedding_dim=3,
            rnn_type='GRU',
            bidirectional=False,
            dropout_rate=0.2
            ):
        super().__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_pain_surveys = num_pain_surveys
        self.embedding_dim = embedding_dim

        # Embedding layers for pain surveys
        self.pain_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_pain_levels, embedding_dim=embedding_dim)
            for _ in range(num_pain_surveys)
        ])
        
        # Total input features after embedding
        total_input_features = num_continuous_features + (num_pain_surveys * embedding_dim)

        # Map string name to PyTorch RNN class
        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }

        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        rnn_module = rnn_map[rnn_type]

        # Dropout is only applied between layers (if num_layers > 1)
        dropout_val = dropout_rate if num_layers > 1 else 0

        # Create the recurrent layer
        self.rnn = rnn_module(
            input_size=total_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        # Calculate input size for the final classifier
        if self.bidirectional:
            classifier_input_size = hidden_size * 2
        else:
            classifier_input_size = hidden_size
        
        # Attention mechanism (optional)
        self.use_attention = True
        if self.use_attention:
            self.attn_W = nn.Linear(classifier_input_size, classifier_input_size, bias=True)
            self.attn_v = nn.Linear(classifier_input_size, 1, bias=False)

        # Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)
    
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

        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_out, hidden = self.rnn(x)

        # Use attention pooling or last hidden state
        if self.use_attention:
            # Attention-based pooling over all timesteps
            context, attn_weights = self.attention(rnn_out)  # (B, H)
        else:
            # LSTM returns (h_n, c_n), we only need h_n
            if self.rnn_type == 'LSTM':
                hidden = hidden[0]

            # hidden shape: (num_layers * num_directions, batch_size, hidden_size)

            if self.bidirectional:
                # Reshape to (num_layers, 2, batch_size, hidden_size)
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)

                # Concat last fwd (hidden[-1, 0, ...]) and bwd (hidden[-1, 1, ...])
                # Final shape: (batch_size, hidden_size * 2)
                context = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
            else:
                # Take the last layer's hidden state
                # Final shape: (batch_size, hidden_size)
                context = hidden[-1]

        # Get logits
        logits = self.classifier(context)
        return logits