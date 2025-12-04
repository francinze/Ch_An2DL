# Model Definitions

This directory contains PyTorch model architectures for time-series pain classification.

## Available Models

### 1. `rnn.py` - RecurrentClassifier
- **Architecture**: Recurrent Neural Network (RNN/LSTM/GRU)
- **Input**: `(batch_size, seq_length, input_size)`
- **Output**: `(batch_size, num_classes)` logits
- **Features**:
  - Supports RNN, LSTM, or GRU variants
  - Optional bidirectional processing
  - Configurable hidden size, layers, and dropout
- **Used in**:
  - `model_training.ipynb` (main training pipeline)
  - `solution_2_binary.ipynb` (binary classification variant)
  - `solution_3_feature_engineering.ipynb` (with enhanced features)

**Example usage:**
```python
from model_definitions.rnn import RecurrentClassifier

model = RecurrentClassifier(
    input_size=30,           # Number of features per timestep
    hidden_size=128,         # Hidden state dimension
    num_layers=2,            # Number of recurrent layers
    num_classes=3,           # Number of output classes
    rnn_type='GRU',          # 'RNN', 'LSTM', or 'GRU'
    bidirectional=False,     # Use bidirectional RNN
    dropout_rate=0.2         # Dropout probability
)
```

### 2. `cnn.py` - CNN1DClassifier
- **Architecture**: 1D Convolutional Neural Network
- **Input**: `(batch_size, seq_length, input_size)`
- **Output**: `(batch_size, num_classes)` logits
- **Features**:
  - Multi-layer 1D convolutions for temporal pattern extraction
  - Batch normalization and dropout for regularization
  - Global average pooling for fixed-size representation
  - Configurable filter sizes and kernel sizes
- **Used in**:
  - `solution_1_cnn_improved.ipynb`

**Example usage:**
```python
from model_definitions.cnn import CNN1DClassifier

model = CNN1DClassifier(
    input_size=30,                    # Number of features per timestep
    num_classes=3,                    # Number of output classes
    num_filters=[64, 128, 256],       # Filters for each conv layer
    kernel_sizes=[5, 5, 3],           # Kernel sizes for each layer
    dropout_rate=0.4                  # Dropout probability
)
```

## Creating a New Model

To add a new model architecture:

1. Create a new Python file in this directory (e.g., `transformer.py`)
2. Define a PyTorch `nn.Module` class
3. Ensure the model accepts `(batch_size, seq_length, input_size)` input
4. Return `(batch_size, num_classes)` logits (raw scores, not probabilities)
5. Import and use it in `model_training.ipynb`

**Template:**
```python
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__()
        # Define layers here
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # ... your forward pass ...
        # return logits of shape: (batch_size, num_classes)
        return logits
```

## Notes on Solution Notebooks

- **solution_2_binary.ipynb**: Uses `RecurrentClassifier` with `num_classes=2` for binary classification (pain vs no-pain)
- **solution_3_feature_engineering.ipynb**: Uses `RecurrentClassifier` with enhanced input features (statistical aggregations, derivatives, etc.)

These solutions demonstrate different **data preparation strategies** rather than different model architectures.
