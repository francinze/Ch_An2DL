"""
CNN with Embedding Layers for Pain Surveys

Implements November 7 Clue: Treat pain surveys as categorical features
that need embeddings, not continuous values.

Architecture:
1. Embedding layers for each pain_survey column (4 total)
2. Concatenate embedded pain surveys with continuous joint features
3. 1D CNN to process the combined time series

Includes:
- CNNWithEmbeddings: Model architecture
- EmbeddingDataset: Custom dataset for categorical + continuous features
- train_epoch_with_embeddings: Training loop
- validate_with_embeddings: Validation loop
"""

import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class CNNWithEmbeddings(nn.Module):
    """
    1D CNN with embedding layers for categorical pain survey features.
    
    Key Innovation:
    - Pain surveys (0-4) are treated as categorical, not continuous
    - Each pain_survey column gets its own embedding layer
    - Embeddings learn semantic relationships between pain levels
    - Concatenated with normalized joint sensor readings before CNN
    """
    
    def __init__(
        self,
        num_continuous_features,  # e.g., 30 joints + 1 prosthetics = 31
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=5,  # Pain levels: 0, 1, 2, 3, 4
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
            embedding_dim: Dimension of embedding vectors (e.g., 3 or 5)
            num_filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.num_pain_surveys = num_pain_surveys
        self.embedding_dim = embedding_dim
        
        # Embedding layers for pain surveys
        # Each pain survey gets its own embedding to learn context-specific patterns
        self.pain_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_pain_levels, embedding_dim=embedding_dim)
            for _ in range(num_pain_surveys)
        ])
        
        # Total input features after embedding:
        # continuous_features + (num_pain_surveys * embedding_dim)
        total_input_features = num_continuous_features + (num_pain_surveys * embedding_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            total_input_features, 
            num_filters[0], 
            kernel_size=kernel_sizes[0], 
            padding=kernel_sizes[0]//2
        )
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        
        self.conv2 = nn.Conv1d(
            num_filters[0], 
            num_filters[1], 
            kernel_size=kernel_sizes[1], 
            padding=kernel_sizes[1]//2
        )
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        
        self.conv3 = nn.Conv1d(
            num_filters[1], 
            num_filters[2], 
            kernel_size=kernel_sizes[2], 
            padding=kernel_sizes[2]//2
        )
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
        Forward pass with separate categorical and continuous inputs.
        
        Args:
            categorical_input: (batch_size, seq_length, num_pain_surveys)
                - Integer values 0-4 representing pain levels
            continuous_input: (batch_size, seq_length, num_continuous_features)
                - Float values representing normalized joint readings + prosthetics
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_length, _ = continuous_input.shape
        
        # Embed each pain survey column separately
        # categorical_input[:, :, i] has shape (batch_size, seq_length)
        embedded_pain_surveys = []
        for i in range(self.num_pain_surveys):
            # Get pain survey column i: (batch_size, seq_length)
            pain_col = categorical_input[:, :, i].long()
            
            # Embed: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
            embedded = self.pain_embeddings[i](pain_col)
            embedded_pain_surveys.append(embedded)
        
        # Concatenate all embedded pain surveys along feature dimension
        # Result: (batch_size, seq_length, num_pain_surveys * embedding_dim)
        embedded_pain = torch.cat(embedded_pain_surveys, dim=2)
        
        # Concatenate embedded pain surveys with continuous features
        # Result: (batch_size, seq_length, total_features)
        x = torch.cat([continuous_input, embedded_pain], dim=2)
        
        # Transpose to (batch_size, features, seq_length) for Conv1d
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
    
    def get_embedding_weights(self, survey_idx):
        """
        Retrieve learned embedding weights for visualization.
        
        Args:
            survey_idx: Index of pain survey (0-3)
        
        Returns:
            Embedding matrix of shape (num_pain_levels, embedding_dim)
        """
        return self.pain_embeddings[survey_idx].weight.data.cpu().numpy()


# Example usage:
if __name__ == "__main__":
    # Test the model
    batch_size = 16
    seq_length = 300
    num_continuous = 31  # 30 joints + 1 prosthetics
    num_pain_surveys = 4
    
    model = CNNWithEmbeddings(
        num_continuous_features=num_continuous,
        num_classes=3,
        num_pain_surveys=4,
        num_pain_levels=5,
        embedding_dim=3,
        dropout_rate=0.4
    )
    
    # Create dummy data
    categorical_data = torch.randint(0, 5, (batch_size, seq_length, num_pain_surveys))
    continuous_data = torch.randn(batch_size, seq_length, num_continuous)
    
    # Forward pass
    output = model(categorical_data, continuous_data)
    
    print("=" * 70)
    print("CNN with Embeddings - Test Forward Pass")
    print("=" * 70)
    print(f"Categorical input shape: {categorical_data.shape}")
    print(f"Continuous input shape: {continuous_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Embedding parameters: {sum(p.numel() for p in model.pain_embeddings.parameters()):,}")
    print("=" * 70)
    
    # Show embedding for pain_survey_1
    print("\nLearned embeddings for pain_survey_1 (before training):")
    print(model.get_embedding_weights(0))


# =============================================================================
# Dataset and Training Functions
# =============================================================================

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Custom dataset for models with separate categorical and continuous inputs.
    
    Used with CNNWithEmbeddings model that requires:
    - categorical_input: pain survey values (integers 0-4) for embedding lookup
    - continuous_input: normalized joint sensors + prosthetics features
    """
    
    def __init__(self, categorical, continuous, labels):
        """
        Args:
            categorical: numpy array of shape (N, seq_length, num_pain_surveys)
            continuous: numpy array of shape (N, seq_length, num_continuous_features)
            labels: numpy array of shape (N,)
        """
        self.categorical = torch.from_numpy(categorical).long()
        self.continuous = torch.from_numpy(continuous).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.categorical[idx], self.continuous[idx], self.labels[idx]
