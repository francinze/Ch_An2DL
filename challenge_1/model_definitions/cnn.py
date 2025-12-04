import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    1D CNN for time-series classification.
    Uses convolutional layers to extract temporal patterns.
    """
    def __init__(
            self,
            input_size,
            num_classes,
            num_filters=[64, 128, 256],
            kernel_sizes=[5, 5, 3],
            dropout_rate=0.4
    ):
        """
        Args:
            input_size: Number of features per timestep
            num_classes: Number of output classes
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        
        self.conv3 = nn.Conv1d(num_filters[1], num_filters[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(num_filters[2])
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(num_filters[2], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv blocks with pooling and dropout
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global pooling to fixed-size representation
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.fc(x)
        return logits
