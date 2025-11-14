import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    1D CNN with embedding layers for categorical pain survey features.
    
    Treats pain surveys (0-4) as categorical, not continuous.
    Each pain_survey column gets its own embedding layer.
    Embeddings are concatenated with normalized joint sensor readings before CNN.
    """
    def __init__(
            self,
            num_continuous_features,
            num_classes=3,
            num_pain_surveys=4,
            num_pain_levels=5,
            embedding_dim=3,
            num_filters=[64, 128, 256],
            kernel_sizes=[5, 5, 3],
            dropout_rate=0.4
    ):
        """
        Args:
            num_continuous_features: Number of continuous features (joints + prosthetics)
            num_classes: Number of output classes (3: no_pain, low_pain, high_pain)
            num_pain_surveys: Number of pain survey columns (4)
            num_pain_levels: Number of pain levels (5: 0-4)
            embedding_dim: Dimension of embedding vectors
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
        """
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
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(total_input_features, num_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
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
