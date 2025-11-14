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
    TCNClassifier:
    - Input:  (batch, seq_len, input_size)
    - Output: (batch, num_classes) logits
    Kwargs:
        channels: List[int] — ampiezza dei layer (es. [64,128,256])
        kernel_size: int — dimensione kernel (es. 5)
        dropout: float — dropout nei blocchi (es. 0.3)
        use_bn: bool — usare BatchNorm nei blocchi
        causal: bool — conv causali (True) o 'same' simmetrica (False)
        gap: str — 'avg' (default) o 'max' per il pooling temporale
    """
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__()
        channels: List[int] = kwargs.get('channels', [64, 128, 256])
        kernel_size: int     = kwargs.get('kernel_size', 5)
        dropout: float       = kwargs.get('dropout', 0.3)
        use_bn: bool         = kwargs.get('use_bn', True)
        causal: bool         = kwargs.get('causal', True)
        gap: str             = kwargs.get('gap', 'avg')

        assert kernel_size >= 2 and kernel_size % 1 == 0, "kernel_size deve essere >=2"
        assert gap in ('avg', 'max'), "gap deve essere 'avg' o 'max'"

        layers = []
        in_ch = input_size  # trattiamo le feature come canali
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
        self.pool = nn.AdaptiveAvgPool1d(1) if gap == 'avg' else nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                  # (B, C, 1) -> (B, C)
            nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        # x: (B, T, F) -> permuta a (B, F, T) per Conv1d
        x = x.transpose(1, 2)
        y = self.tcn(x)             # (B, C, T)
        y = self.pool(y)            # (B, C, 1)
        logits = self.head(y)       # (B, num_classes)
        return logits
