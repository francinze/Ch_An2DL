import torch.nn as nn
from typing import List

class _TemporalBlock(nn.Module):
    """
    Residual block: Conv1d (dilated, causal) -> Norm -> PReLU -> Dropout -> Conv1d -> Norm -> PReLU
    con skip connection + proiezione 1x1 se cambia il numero di canali.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2, use_bn=True, causal=True):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        # causal padding: pad left-only; gestito in forward con F.pad
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, bias=not use_bn)
        self.norm1 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act1  = nn.PReLU(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=0, dilation=dilation, bias=not use_bn)
        self.norm2 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act2  = nn.PReLU(out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def _causal_pad(self, x):
        # Pad only on the left to preserve causality
        pad = (self.kernel_size - 1) * self.dilation
        if pad > 0:
            return nn.functional.pad(x, (pad, 0))
        return x

    def forward(self, x):
        # x: (B, C, T)
        y = self._causal_pad(x) if self.causal else x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self._causal_pad(y) if self.causal else y
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        s = self.downsample(x)
        return y + s  # residual


class TCN(nn.Module):
    """
    TCN Classifier with embedding layers for categorical pain survey features.
    
    - Input:  (batch, seq_len, num_continuous_features) and (batch, seq_len, num_pain_surveys)
    - Output: (batch, num_classes) logits
    
    Treats pain surveys (0-4) as categorical, not continuous.
    Each pain_survey column gets its own embedding layer.
    Embeddings are concatenated with continuous features before TCN.
    """
    def __init__(self, num_continuous_features, num_classes=3, num_pain_surveys=4, 
                 num_pain_levels=5, embedding_dim=3, **kwargs):
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
        
        channels: List[int] = kwargs.get('channels', [64, 128, 256])
        kernel_size: int     = kwargs.get('kernel_size', 5)
        dropout: float       = kwargs.get('dropout', 0.3)
        use_bn: bool         = kwargs.get('use_bn', True)
        causal: bool         = kwargs.get('causal', True)
        gap: str             = kwargs.get('gap', 'avg')

        assert kernel_size >= 2 and kernel_size % 1 == 0, "kernel_size deve essere >=2"
        assert gap in ('avg', 'max'), "gap deve essere 'avg' o 'max'"

        layers = []
        in_ch = total_input_features  # use total features after embedding
        # dilations: 1,2,4,8,... per ogni blocco
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                _TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_bn=use_bn,
                    causal=causal
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        
        # Attention mechanism (optional)
        self.use_attention = True
        if self.use_attention:
            self.attn_W = nn.Linear(channels[-1], channels[-1], bias=True)
            self.attn_v = nn.Linear(channels[-1], 1, bias=False)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1) if gap == 'avg' else nn.AdaptiveMaxPool1d(1)
        
        self.head = nn.Sequential(
            nn.Flatten() if not self.use_attention else nn.Identity(),
            nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )
    
    def attention(self, tcn_out):
        """
        Bahdanau-style attention over temporal dimension.
        
        Args:
            tcn_out: (batch_size, channels, seq_len)
        
        Returns:
            context: (batch_size, channels)
            attn_weights: (batch_size, seq_len)
        """
        import torch
        # Transpose to (batch, seq_len, channels)
        x = tcn_out.transpose(1, 2)
        
        # u_t = tanh(W * h_t)
        u = torch.tanh(self.attn_W(x))  # (B, T, C)
        
        # score_t = v^T u_t
        scores = self.attn_v(u).squeeze(-1)  # (B, T)
        
        # α_t = softmax(scores)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (B, T)
        
        # context = Σ_t α_t * h_t
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, C)
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
        
        # x: (B, T, F) -> permuta a (B, F, T) per Conv1d
        x = x.transpose(1, 2)
        y = self.tcn(x)             # (B, C, T)
        
        # Temporal pooling
        if self.use_attention:
            # Attention-based pooling
            context, attn_weights = self.attention(y)  # (B, C)
        else:
            # Global pooling
            context = self.pool(y).squeeze(-1)  # (B, C)
        
        logits = self.head(context)  # (B, num_classes)
        return logits
