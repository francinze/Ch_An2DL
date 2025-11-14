import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN + LSTM with embedding layers for categorical pain survey features.
    
    Architecture:
        Embeddings for pain surveys
        → Conv1D blocks (local pattern extraction)
        → LSTM layers (temporal reasoning)
        → Attention mechanism (optional)
        → FC layers → Softmax
    """
    
    def __init__(
        self,
        num_continuous_features,
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=3,
        embedding_dim=3,
        cnn_filters=[64, 128],
        cnn_kernel_sizes=[5, 5],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout_rate=0.3,
        bidirectional=True,
        use_attention=True,
        **kwargs
    ):
        """
        Args:
            num_continuous_features: Number of continuous features (joints + prosthetics)
            num_classes: Number of output classes
            num_pain_surveys: Number of pain survey columns (4)
            num_pain_levels: Number of pain levels (5: 0-4)
            embedding_dim: Dimension of embedding vectors
            cnn_filters: List of filter counts for each conv layer
            cnn_kernel_sizes: List of kernel sizes for each conv layer
            lstm_hidden_size: Hidden dimension of LSTM
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism for pooling
        """
        super().__init__()
        
        self.num_continuous_features = num_continuous_features
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
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
        # 1) CNN Feature Extractor
        # -----------------------
        cnn_layers = []
        in_channels = total_input_features
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_filters[-1]
        
        # -----------------------
        # 2) LSTM for Temporal Modeling
        # -----------------------
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0
        )
        
        lstm_out_dim = lstm_hidden_size * (2 if bidirectional else 1)
        
        # -----------------------
        # 3) Attention Mechanism (optional)
        # -----------------------
        if use_attention:
            self.attn_W = nn.Linear(lstm_out_dim, lstm_out_dim, bias=True)
            self.attn_v = nn.Linear(lstm_out_dim, 1, bias=False)
        
        # -----------------------
        # 4) Classification Head
        # -----------------------
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim // 2, num_classes)
        )
    
    def attention(self, lstm_out):
        """
        Bahdanau-style attention mechanism.
        
        Args:
            lstm_out: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: (batch_size, hidden_dim)
            attn_weights: (batch_size, seq_len)
        """
        # u_t = tanh(W_h * h_t)
        u = torch.tanh(self.attn_W(lstm_out))  # (B, T, D)
        
        # score_t = v^T u_t
        scores = self.attn_v(u).squeeze(-1)  # (B, T)
        
        # α_t = softmax(scores)
        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        
        # context = Σ_t α_t * h_t
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
        
        # 1) CNN Feature Extraction
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)  # (B, F, T)
        
        # Apply convolutional layers
        cnn_out = self.cnn(x)  # (B, C, T')
        
        # Transpose back to (batch, seq_len, features) for LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (B, T', C)
        
        # 2) LSTM Temporal Modeling
        lstm_out, _ = self.lstm(cnn_out)  # (B, T', H)
        
        # 3) Pooling across time dimension
        if self.use_attention:
            # Attention-based pooling
            context, attn_weights = self.attention(lstm_out)  # (B, H)
        else:
            # Mean pooling
            context = torch.mean(lstm_out, dim=1)  # (B, H)
        
        # 4) Classification
        context = self.dropout(context)
        logits = self.classifier(context)  # (B, num_classes)
        
        return logits


class CNNGRUClassifier(nn.Module):
    """
    Hybrid CNN + GRU with embedding layers.
    
    Similar to CNNLSTMClassifier but uses GRU instead of LSTM.
    GRU is often faster and requires less memory than LSTM.
    """
    
    def __init__(
        self,
        num_continuous_features,
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=3,
        embedding_dim=3,
        cnn_filters=[64, 128],
        cnn_kernel_sizes=[5, 5],
        gru_hidden_size=128,
        gru_num_layers=2,
        dropout_rate=0.3,
        bidirectional=True,
        use_attention=True,
        **kwargs
    ):
        super().__init__()
        
        self.num_continuous_features = num_continuous_features
        self.num_classes = num_classes
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
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
        # 1) CNN Feature Extractor
        # -----------------------
        cnn_layers = []
        in_channels = total_input_features
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = cnn_filters[-1]
        
        # -----------------------
        # 2) GRU for Temporal Modeling
        # -----------------------
        self.gru = nn.GRU(
            input_size=self.cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if gru_num_layers > 1 else 0.0
        )
        
        gru_out_dim = gru_hidden_size * (2 if bidirectional else 1)
        
        # -----------------------
        # 3) Attention Mechanism (optional)
        # -----------------------
        if use_attention:
            self.attn_W = nn.Linear(gru_out_dim, gru_out_dim, bias=True)
            self.attn_v = nn.Linear(gru_out_dim, 1, bias=False)
        
        # -----------------------
        # 4) Classification Head
        # -----------------------
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_out_dim // 2, num_classes)
        )
    
    def attention(self, gru_out):
        """
        Bahdanau-style attention mechanism.
        
        Args:
            gru_out: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: (batch_size, hidden_dim)
            attn_weights: (batch_size, seq_len)
        """
        u = torch.tanh(self.attn_W(gru_out))
        scores = self.attn_v(u).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)
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
        
        # 1) CNN Feature Extraction
        x = x.transpose(1, 2)  # (B, F, T)
        cnn_out = self.cnn(x)  # (B, C, T')
        cnn_out = cnn_out.transpose(1, 2)  # (B, T', C)
        
        # 2) GRU Temporal Modeling
        gru_out, _ = self.gru(cnn_out)  # (B, T', H)
        
        # 3) Pooling across time dimension
        if self.use_attention:
            context, attn_weights = self.attention(gru_out)
        else:
            context = torch.mean(gru_out, dim=1)
        
        # 4) Classification
        context = self.dropout(context)
        logits = self.classifier(context)
        
        return logits
