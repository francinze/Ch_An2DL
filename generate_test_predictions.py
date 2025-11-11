"""
Generate Test Predictions for Competition Submission

This script loads the best CNN model and generates predictions for the test set.
Use this to validate if the high validation F1 translates to good test performance.
"""

SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import random
import numpy as np
np.random.seed(SEED)
random.seed(SEED)

import torch
torch.manual_seed(SEED)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
else:
    device = torch.device("cpu")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print(f"Device: {device}")

# -------------------------------------------------------------------
# 1. Load and preprocess test data (same as training)
# -------------------------------------------------------------------
print("\nLoading test data...")
df_test = pd.read_csv("pirate_pain_test.csv")

number_cols = ['n_legs', 'n_hands', 'n_eyes']
for col in number_cols:
    df_test[col] = df_test[col].astype('category').cat.codes

joint_cols = ["joint_" + str(i).zfill(2) for i in range(31)]
for col in joint_cols:
    df_test[col] = df_test[col].astype(np.float32)

# IMPORTANT: Use same scaler fit on training data
# If you saved scaler: minmax_scaler = pickle.load(open('scaler.pkl', 'rb'))
# For now, re-fit (ideally save scaler during training)
print("WARNING: Refitting scaler on test data (ideally load saved training scaler)")
minmax_scaler = MinMaxScaler()
df_test[joint_cols] = minmax_scaler.fit_transform(df_test[joint_cols])

data_cols = number_cols + joint_cols

print(f"Test samples: {len(df_test['sample_index'].unique())}")

# -------------------------------------------------------------------
# 2. Build test sequences (use SAME parameters as training)
# -------------------------------------------------------------------
WINDOW_SIZE = 300
STRIDE = 150  # Use same stride as improved model

def build_test_sequences(df, window=300, stride=150):
    """Build sequences for test set - returns sequences and sample indices"""
    dataset = []
    sample_indices = []
    
    for id in df['sample_index'].unique():
        temp = df[df['sample_index'] == id][data_cols].values
        
        padding_len = window - len(temp) % window
        padding = np.zeros((padding_len, len(data_cols)), dtype='float32')
        temp = np.concatenate((temp, padding))
        
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            sample_indices.append(id)  # Track which sample this belongs to
            idx += stride
    
    return np.array(dataset), np.array(sample_indices)

X_test, test_sample_ids = build_test_sequences(df_test, WINDOW_SIZE, STRIDE)
print(f"\nTest sequences: {X_test.shape}")
print(f"Sequences per sample: ~{len(X_test) / len(df_test['sample_index'].unique()):.1f}")


# -------------------------------------------------------------------
# 3. Define model architecture (MUST match training)
# -------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# -------------------------------------------------------------------
# 4. Load trained model
# -------------------------------------------------------------------
print("\nLoading model...")
model = CNN1DClassifier(input_size=X_test.shape[-1], num_classes=3, dropout=0.4).to(device)

# Try loading improved model first, fall back to original
try:
    model.load_state_dict(torch.load('models/cnn_improved_best.pt', map_location=device))
    print("✅ Loaded IMPROVED CNN model (with stratified split)")
    model_name = "improved"
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load('models/cnn_best.pt', map_location=device))
        print("⚠️  Loaded ORIGINAL CNN model (without stratification)")
        model_name = "original"
    except FileNotFoundError:
        print("❌ ERROR: No trained model found!")
        print("   Run solution_1_cnn_improved.ipynb or solution_1_cnn.ipynb first")
        exit(1)

model.eval()

# -------------------------------------------------------------------
# 5. Generate predictions
# -------------------------------------------------------------------
print("\nGenerating predictions...")
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test).float()), 
    batch_size=32, 
    shuffle=False
)

all_preds = []
all_probs = []

with torch.no_grad():
    for (inputs,) in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print(f"Generated {len(all_preds)} sequence-level predictions")

# -------------------------------------------------------------------
# 6. Aggregate predictions per sample (majority vote)
# -------------------------------------------------------------------
print("\nAggregating predictions per sample...")

# Option 1: Majority vote
sample_predictions = {}
for sample_id, pred in zip(test_sample_ids, all_preds):
    if sample_id not in sample_predictions:
        sample_predictions[sample_id] = []
    sample_predictions[sample_id].append(pred)

# Take majority vote for each sample
final_predictions = {}
for sample_id, preds in sample_predictions.items():
    # Majority vote
    unique, counts = np.unique(preds, return_counts=True)
    final_predictions[sample_id] = unique[np.argmax(counts)]

# Option 2: Average probabilities (alternative)
sample_probs = {}
for sample_id, prob in zip(test_sample_ids, all_probs):
    if sample_id not in sample_probs:
        sample_probs[sample_id] = []
    sample_probs[sample_id].append(prob)

final_predictions_avgprob = {}
for sample_id, probs in sample_probs.items():
    avg_prob = np.mean(probs, axis=0)
    final_predictions_avgprob[sample_id] = np.argmax(avg_prob)

# Compare both methods
agreement = sum(final_predictions[sid] == final_predictions_avgprob[sid] for sid in final_predictions)
print(f"Agreement between majority vote and avg probability: {agreement}/{len(final_predictions)} ({agreement/len(final_predictions)*100:.1f}%)")

# -------------------------------------------------------------------
# 7. Create submission file
# -------------------------------------------------------------------
label_mapping_inverse = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}

# Using majority vote (Option 1)
submission_df = pd.DataFrame({
    'sample_index': list(final_predictions.keys()),
    'label': [label_mapping_inverse[pred] for pred in final_predictions.values()]
})
submission_df = submission_df.sort_values('sample_index')

submission_file = f'submission_{model_name}_majority_vote.csv'
submission_df.to_csv(submission_file, index=False)
print(f"\n✅ Submission file created: {submission_file}")

# Using average probability (Option 2)
submission_df_avgprob = pd.DataFrame({
    'sample_index': list(final_predictions_avgprob.keys()),
    'label': [label_mapping_inverse[pred] for pred in final_predictions_avgprob.values()]
})
submission_df_avgprob = submission_df_avgprob.sort_values('sample_index')

submission_file_avgprob = f'submission_{model_name}_avg_prob.csv'
submission_df_avgprob.to_csv(submission_file_avgprob, index=False)
print(f"✅ Submission file created: {submission_file_avgprob}")

# -------------------------------------------------------------------
# 8. Show prediction distribution
# -------------------------------------------------------------------
print(f"\n📊 Test Prediction Distribution (Majority Vote):")
for cls, label in label_mapping_inverse.items():
    count = sum(1 for pred in final_predictions.values() if pred == cls)
    print(f"  {label}: {count} ({count/len(final_predictions)*100:.1f}%)")

print(f"\n📊 Test Prediction Distribution (Avg Probability):")
for cls, label in label_mapping_inverse.items():
    count = sum(1 for pred in final_predictions_avgprob.values() if pred == cls)
    print(f"  {label}: {count} ({count/len(final_predictions_avgprob)*100:.1f}%)")

print("\n" + "=" * 80)
print("🚀 Ready to submit!")
print("=" * 80)
print(f"\nUpload '{submission_file}' or '{submission_file_avgprob}' to competition")
print("\n💡 Tips:")
print("   - If test accuracy is much lower than validation F1:")
print("     → Train/val split doesn't match test distribution")
print("     → Use solution_1_cnn_improved.ipynb with stratified split")
print("   - If predictions are mostly one class:")
print("     → Check test data distribution")
print("     → May need different aggregation strategy")
