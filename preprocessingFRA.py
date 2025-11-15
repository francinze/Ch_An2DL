# Set seed for reproducibility
SEED = 42

# Import necessary libraries
import os

from preprocessing_embedding import add_time_features

# Set environment variables before importing modules
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(SEED)
random.seed(SEED)

# Import PyTorch
import torch
torch.manual_seed(SEED)
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

# Import other libraries
import pandas as pd

# Data Loading

df = pd.read_csv("pirate_pain_train.csv")
df_test = pd.read_csv("pirate_pain_test.csv")
df = df.drop(columns=['joint_30'])
df_test = df_test.drop(columns=['joint_30'])
df = df_test.drop(columns=['joint_11'])
df_test = df_test.drop(columns=['joint_11'])

print("Training data shape:", df.shape)

def add_prosthetics_feature(df, df_test):
    # Create binary 'has_prosthetics' feature (0 = all natural, 1 = has prosthetics)
    print("Creating consolidated feature: 'has_prosthetics'")
    print("=" * 60)

    # Create the new feature
    df['has_prosthetics'] = (df['n_legs'] != 'two').astype(int)
    df_test['has_prosthetics'] = (df_test['n_legs'] != 'two').astype(int)

    # Show the mapping
    print("\nMapping:")
    print("  has_prosthetics = 0 → All natural body parts (two legs, two hands, two eyes)")
    print("  has_prosthetics = 1 → Has prosthetics (peg leg, hook hand, eye patch)")

    # Show distribution
    print("\n" + "=" * 60)
    print("Distribution of new feature:")
    print("=" * 60)
    print("\nTraining set:")
    train_dist = df['has_prosthetics'].value_counts().sort_index()
    for value, count in train_dist.items():
        label = "Natural" if value == 0 else "Prosthetics"
        pct = (count / len(df)) * 100
        print(f"  {value} ({label:12s}): {count:6,} samples ({pct:.2f}%)")

    print("\nTest set:")
    test_dist = df_test['has_prosthetics'].value_counts().sort_index()
    for value, count in test_dist.items():
        label = "Natural" if value == 0 else "Prosthetics"
        pct = (count / len(df_test)) * 100
        print(f"  {value} ({label:12s}): {count:6,} samples ({pct:.2f}%)")


    # Columns to drop
    cols_to_drop = ['n_legs', 'n_hands', 'n_eyes', 
                    'n_legs_encoded', 'n_hands_encoded', 'n_eyes_encoded']

    # Drop from both train and test
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])

    print("\nFeature created successfully!")
    return df, df_test

def scale_joint_columns(df, use_existing_scaler=None):
    print("\nApplying Min-Max normalization to joint columns...")
    print("=" * 60)
    # List of joint columns to normalize
    all_joint_cols = ["joint_" + str(i).zfill(2) for i in range(30)]

    # Filter to include only existing joint columns in the DataFrame
    joint_cols = [col for col in all_joint_cols if col in df.columns]

    if not joint_cols:
        print("No relevant joint columns found in the DataFrame to scale. Skipping scaling.")
        return df

    for col in joint_cols:
        df[col] = df[col].astype(np.float32)

    # Save the fitted scaler for later use on test data
    import pickle
    if not use_existing_scaler:
        # Initialize the MinMaxScaler
        minmax_scaler = MinMaxScaler()

        # Apply Min-Max normalization to the joint columns
        df[joint_cols] = minmax_scaler.fit_transform(df[joint_cols])


        # Save the scaler that was fitted on training data
        with open('minmax_scaler.pkl', 'wb') as f:
            pickle.dump(minmax_scaler, f)
    else:
        # Load the existing scaler
        minmax_scaler = pickle.load(open('minmax_scaler.pkl', 'rb'))

        # Apply the existing scaler to the joint columns
        df[joint_cols] = minmax_scaler.transform(df[joint_cols])

    print("✅ Scaler saved successfully!")
    print(f"Scaler learned from training data - Min: {minmax_scaler.data_min_[:5]}")
    print(f"Scaler learned from training data - Max: {minmax_scaler.data_max_[:5]}")
    return df

def apply_target_weighting(target):

    # Define Weights
    WEIGHTS = []
    for label in np.unique(target['label']):
        print(f"Label: {label}, Count: {len(target[target['label'] == label])}")
        WEIGHTS.append(len(target) / len(target[target['label'] == label]))
    WEIGHTS = torch.Tensor(WEIGHTS).to(device)

    # Define a mapping of pain indexes to integer labels
    label_mapping = {
        'no_pain': 0,
        'low_pain': 1,
        'high_pain': 2
    }

    # Map pain indexes to integers
    target['label'] = target['label'].map(label_mapping)

    return target

# Data Preprocessing

def train_val_split(df, target, val_ratio=0.2):
    print("\nPerforming train/validation split based on unique users...")
    print("=" * 60)
    # Get unique user IDs and shuffle them
    unique_users = df['sample_index'].unique()
    random.seed(SEED) # Ensure reproducibility of shuffling
    random.shuffle(unique_users)
    input_shape = df.shape

    print(f"Input shape: {input_shape}")

    # Determine the number of users for validation
    num_val_users = int(len(unique_users) * val_ratio)
    val_users = unique_users[:num_val_users]
    train_users = unique_users[num_val_users:]
    print(f"Number of training users: {len(train_users)}")
    print(f"Number of validation users: {len(val_users)}")
    # Split the DataFrame and target based on user IDs
    train_df = df[df['sample_index'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['sample_index'].isin(val_users)].reset_index(drop=True)
    train_target = target[target['sample_index'].isin(train_users)].reset_index(drop=True)  
    val_target = target[target['sample_index'].isin(val_users)].reset_index(drop=True)
    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    return train_df, val_df, train_target, val_target

# Define a function to build sequences from the dataset
def build_sequences(df, window=200, stride=200):
    # Sanity check to ensure the window is divisible by the stride
    assert window % stride == 0

    # Initialise lists to store sequences and their corresponding labels
    dataset = []
    labels = []

    # Iterate over unique IDs in the DataFrame
    for id in df['sample_index'].unique():
        # Extract sensor data for the current ID
        temp = df[df['sample_index'] == id][data_cols].values

        # Retrieve the activity label for the current ID
        label = target[target['sample_index'] == id]['label'].values[0]

        # Calculate padding length to ensure full windows
        padding_len = window - len(temp) % window

        # Create zero padding and concatenate with the data
        padding = np.zeros((padding_len, len(data_cols)), dtype='float32')
        temp = np.concatenate((temp, padding))

        # Build feature windows and associate them with labels
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            labels.append(label)
            idx += stride

    # Convert lists to numpy arrays for further processing
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def build_test_sequences(df, window=200, stride=200):
    # Sanity check to ensure the window is divisible by the stride
    assert window % stride == 0

    # Initialise lists to store sequences and their corresponding labels
    dataset = []

    # Iterate over unique IDs in the DataFrame
    for id in df['sample_index'].unique():
        # Extract sensor data for the current ID
        temp = df[df['sample_index'] == id][data_cols].values

        # Calculate padding length to ensure full windows
        padding_len = window - len(temp) % window

        # Create zero padding and concatenate with the data
        padding = np.zeros((padding_len, len(data_cols)), dtype='float32')
        temp = np.concatenate((temp, padding))

        # Build feature windows
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            idx += stride

    # Convert lists to numpy arrays for further processing
    dataset = np.array(dataset)

    return dataset

# joint diversi tra train e test
# da 13 a 17 da 19 a 25

# {'window': 256, 'stride': 64, 'labeling': 'id', 'padding': 'zero'} -> (np.float64(0.5744914366114311), np.float64(0.05140432771327308), [0.6315323565323566, 0.528096416254311, 0.6086213303604607, 0.4990065786568944, 0.6052005012531328])
# 
# Questo è il milgiore

def run_preprocessing():
    df = pd.read_csv("pirate_pain_train.csv")
    df_test = pd.read_csv("pirate_pain_test.csv")
    df = df.drop(columns=['joint_30'])
    df_test = df_test.drop(columns=['joint_30'])
    df = df.drop(columns=['joint_11'])
    df_test = df_test.drop(columns=['joint_11'])
    
    # Target
    target = pd.read_csv("pirate_pain_train_labels.csv")
    target.head()


    # Add time-based features (November 12 clue implementation)
    df, df_test = add_time_features(df, df_test)
    df, df_test = add_prosthetics_feature(df, df_test)
    df = scale_joint_columns(df)
    df_test = scale_joint_columns(df_test)
    target = apply_target_weighting(target)
    train_df, val_df, train_target, val_target = train_val_split(df, target, val_ratio=0.2)

    return train_df, val_df, train_target, val_target, df_test