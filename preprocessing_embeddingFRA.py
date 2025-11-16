# Set seed for reproducibility
SEED = 42

# Import necessary libraries
import os

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
df = df.drop(columns=['time'])
df_test = df_test.drop(columns=['time'])

print("Training data shape:", df.shape)

def add_time_features(df, df_test):
    """
    Add time-based features implementing November 12 clue:
    'Not only what happens, but when. Time, not just an index, but a feature it is.'
    
    Treats the 'time' column as a rich feature rather than ignoring it.
    Creates both normalized position and cyclical encodings.
    """
    print("\nCreating time-based features from 'time' column")
    print("=" * 60)
    
    # Feature 1: Normalized time (position in sequence: 0.0 to 1.0)
    print("\n1. Normalized Time (relative position in sequence)")
    df['time_normalized'] = df.groupby('sample_index')['time'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    df_test['time_normalized'] = df_test.groupby('sample_index')['time'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    
    # Analyze sequence lengths to determine cyclical period
    train_lengths = df.groupby('sample_index')['time'].max()
    test_lengths = df_test.groupby('sample_index')['time'].max()
    avg_length = train_lengths.mean()
    
    print(f"   - Average sequence length: {avg_length:.1f} timesteps")
    print(f"   - Train range: {train_lengths.min():.0f} to {train_lengths.max():.0f}")
    print(f"   - Test range: {test_lengths.min():.0f} to {test_lengths.max():.0f}")
    
    # Feature 2: Cyclical encoding (captures periodic patterns)
    # Use a period based on average sequence length for meaningful cycles
    period = max(50, avg_length / 3)  # Create ~3 cycles per sequence
    print(f"\n2. Cyclical Encoding (period={period:.1f} timesteps)")
    print(f"   - Captures repeating patterns within sequences")
    
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / period)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / period)
    df_test['time_sin'] = np.sin(2 * np.pi * df_test['time'] / period)
    df_test['time_cos'] = np.cos(2 * np.pi * df_test['time'] / period)
    
    # Feature 3: Time position categories (early/mid/late)
    print("\n3. Time Position Category (early/mid/late in sequence)")
    
    def categorize_time_position(group):
        normalized = group / group.max() if group.max() > 0 else 0
        return pd.cut(normalized, bins=[0, 0.33, 0.66, 1.0], 
                     labels=[0, 1, 2], include_lowest=True).astype(int)
    
    df['time_position'] = df.groupby('sample_index')['time'].transform(categorize_time_position)
    df_test['time_position'] = df_test.groupby('sample_index')['time'].transform(categorize_time_position)
    
    print("   - 0: Early (0-33% of sequence)")
    print("   - 1: Mid (33-66% of sequence)")
    print("   - 2: Late (66-100% of sequence)")
    
    # Show distribution of time position categories
    print("\n" + "=" * 60)
    print("Distribution of time position categories:")
    print("=" * 60)
    print("\nTraining set:")
    train_dist = df['time_position'].value_counts().sort_index()
    for value, count in train_dist.items():
        label = ['Early', 'Mid', 'Late'][value]
        pct = (count / len(df)) * 100
        print(f"  {value} ({label:5s}): {count:6,} samples ({pct:.2f}%)")
    
    print("\nTest set:")
    test_dist = df_test['time_position'].value_counts().sort_index()
    for value, count in test_dist.items():
        label = ['Early', 'Mid', 'Late'][value]
        pct = (count / len(df_test)) * 100
        print(f"  {value} ({label:5s}): {count:6,} samples ({pct:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Summary: Created 4 new time features")
    print("=" * 60)
    print("  ✅ time_normalized: Continuous [0.0, 1.0] - position in sequence")
    print("  ✅ time_sin: Continuous [-1.0, 1.0] - cyclical encoding")
    print("  ✅ time_cos: Continuous [-1.0, 1.0] - cyclical encoding")
    print("  ✅ time_position: Categorical [0, 1, 2] - early/mid/late (for embeddings)")
    print("\n  Note: Original 'time' column preserved for reference")
    print("=" * 60)
    
    return df, df_test

def add_prosthetics_feature(df, df_test):
    # Create binary 'has_prosthetics' feature (0 = all natural, 1 = has prosthetics)
    print("\nCreating consolidated feature: 'has_prosthetics'")
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

def prepare_data_with_embeddings(df, df_test):
    """
    Prepare data for models that use embeddings for pain surveys and time position.
    
    Returns separate arrays for:
    - Categorical features (pain surveys + time_position): kept as integers for embedding lookup
    - Continuous features (joints + prosthetics + time features): normalized floats
    
    This implements:
    - November 7 clue: treat pain surveys as categorical features that need embeddings
    - November 12 clue: use time as a rich feature (normalized, cyclical, categorical)
    """
    # Pain survey columns (categorical: 0-4)
    pain_survey_cols = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4']
    
    # Time categorical column (0-2: early/mid/late)
    time_categorical_cols = ['time_position']
    
    # All categorical columns for embeddings
    categorical_cols = pain_survey_cols + time_categorical_cols
    
    # Joint columns (continuous)
    joint_cols = ["joint_" + str(i).zfill(2) for i in range(30)]
    
    # Time continuous columns (normalized + cyclical)
    time_continuous_cols = ['time_normalized', 'time_sin', 'time_cos']
    
    print("=" * 70)
    print("Preparing data for EMBEDDING-based models with TIME FEATURES")
    print("=" * 70)
    print(f"\n✅ Categorical features (for embeddings):")
    print(f"   - Pain surveys: {pain_survey_cols}")
    print(f"     → nn.Embedding(num_embeddings=5, embedding_dim=3)")
    print(f"   - Time position: {time_categorical_cols}")
    print(f"     → nn.Embedding(num_embeddings=3, embedding_dim=2)")
    print(f"\n✅ Continuous features: {len(joint_cols)} joints + {len(time_continuous_cols)} time + has_prosthetics")
    print(f"   - Joint sensors: {len(joint_cols)} features (normalized)")
    print(f"   - Time features: {time_continuous_cols}")
    print(f"   - Body feature: has_prosthetics")
    print("=" * 70)
    
    # Verify categorical features are integers in valid range
    for col in pain_survey_cols:
        train_unique = df[col].unique()
        test_unique = df_test[col].unique()
        print(f"\n{col}: train={sorted(train_unique)}, test={sorted(test_unique)}")
        
        # Ensure they are integers
        df[col] = df[col].astype(np.int64)
        df_test[col] = df_test[col].astype(np.int64)
    
    # Verify time_position is integer
    for col in time_categorical_cols:
        train_unique = df[col].unique()
        test_unique = df_test[col].unique()
        print(f"\n{col}: train={sorted(train_unique)}, test={sorted(test_unique)}")
        df[col] = df[col].astype(np.int64)
        df_test[col] = df_test[col].astype(np.int64)
    
    return df, df_test, categorical_cols, joint_cols + time_continuous_cols + ['has_prosthetics']

def build_sequences_with_embeddings(df, target, window=200, stride=200, 
                                   pain_survey_cols=None, continuous_cols=None):
    """
    Build sequences separating categorical (pain surveys) from continuous features.
    
    Returns:
        categorical_sequences: (N, window, 4) - pain survey values for embedding
        continuous_sequences: (N, window, num_continuous) - normalized joint + prosthetics
        labels: (N,) - pain labels
    """
    if pain_survey_cols is None:
        pain_survey_cols = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4']
    
    if continuous_cols is None:
        # Joints + has_prosthetics
        joint_cols = ["joint_" + str(i).zfill(2) for i in range(30)]
        continuous_cols = joint_cols + ['has_prosthetics']
    
    # Ensure window is divisible by stride
    assert window % stride == 0
    
    categorical_data = []
    continuous_data = []
    labels = []
    
    for id in df['sample_index'].unique():
        # Extract categorical features (pain surveys)
        cat_temp = df[df['sample_index'] == id][pain_survey_cols].values
        
        # Extract continuous features (joints + prosthetics)
        cont_temp = df[df['sample_index'] == id][continuous_cols].values
        
        # Get label
        label = target[target['sample_index'] == id]['label'].values[0]
        
        # Calculate padding
        padding_len = window - len(cat_temp) % window
        
        # Pad categorical data
        cat_padding = np.zeros((padding_len, len(pain_survey_cols)), dtype='int64')
        cat_temp = np.concatenate((cat_temp, cat_padding))
        
        # Pad continuous data
        cont_padding = np.zeros((padding_len, len(continuous_cols)), dtype='float32')
        cont_temp = np.concatenate((cont_temp, cont_padding))
        
        # Build windows
        idx = 0
        while idx + window <= len(cat_temp):
            categorical_data.append(cat_temp[idx:idx + window])
            continuous_data.append(cont_temp[idx:idx + window])
            labels.append(label)
            idx += stride
    
    categorical_data = np.array(categorical_data)
    continuous_data = np.array(continuous_data)
    labels = np.array(labels)
    
    print(f"\n📊 Sequences built:")
    print(f"   Categorical shape: {categorical_data.shape} (for embeddings)")
    print(f"   Continuous shape: {continuous_data.shape} (joints + prosthetics)")
    print(f"   Labels shape: {labels.shape}")
    
    return categorical_data, continuous_data, labels

def build_test_sequences_with_embeddings(df, window=200, stride=200,
                                        pain_survey_cols=None, continuous_cols=None):
    """
    Build test sequences separating categorical from continuous features.
    """
    if pain_survey_cols is None:
        pain_survey_cols = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4']
    
    if continuous_cols is None:
        joint_cols = ["joint_" + str(i).zfill(2) for i in range(30)]
        continuous_cols = joint_cols + ['has_prosthetics']
    
    assert window % stride == 0
    
    categorical_data = []
    continuous_data = []
    
    for id in df['sample_index'].unique():
        cat_temp = df[df['sample_index'] == id][pain_survey_cols].values
        cont_temp = df[df['sample_index'] == id][continuous_cols].values
        
        padding_len = window - len(cat_temp) % window
        
        cat_padding = np.zeros((padding_len, len(pain_survey_cols)), dtype='int64')
        cat_temp = np.concatenate((cat_temp, cat_padding))
        
        cont_padding = np.zeros((padding_len, len(continuous_cols)), dtype='float32')
        cont_temp = np.concatenate((cont_temp, cont_padding))
        
        idx = 0
        while idx + window <= len(cat_temp):
            categorical_data.append(cat_temp[idx:idx + window])
            continuous_data.append(cont_temp[idx:idx + window])
            idx += stride
    
    categorical_data = np.array(categorical_data)
    continuous_data = np.array(continuous_data)
    
    return categorical_data, continuous_data

def run_preprocessing():
    """
    Complete preprocessing pipeline for embedding-based models.
    
    Returns all necessary data for training:
    - Categorical features (pain surveys + time_position) for embeddings
    - Continuous features (joints + prosthetics + time features)
    - Column metadata for building sequences
    """
    print("\n" + "="*70)
    print("RUNNING COMPLETE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load data
    df = pd.read_csv("pirate_pain_train.csv")
    df_test = pd.read_csv("pirate_pain_test.csv")
    
    # Load target
    target = pd.read_csv("pirate_pain_train_labels.csv")
    
    # *************************************************************
    # PASSO 1: ADD TIME FEATURES (RICHIEDE ANCORA LA COLONNA 'time')
    # *************************************************************
    df, df_test = add_time_features(df, df_test)
    
    # *************************************************************
    # PASSO 2: DROP COLONNE 
    # *************************************************************
    # Colonna 'time' (usata per le feature), 'joint_30' (zero varianza)
    # e 'joint_11' (rimossa per allineare train/test o per altro motivo).
    cols_to_drop = ['joint_30', 'joint_11', 'time']
    
    # Correzione del probabile errore di battitura nella riga che seguiva
    # la rimozione di joint_30. Assumo l'intento di eliminare joint_11 da entrambi:
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])
    
    # Add prosthetics feature (usa n_legs, n_hands, n_eyes)
    df, df_test = add_prosthetics_feature(df, df_test)
    
    # Scale joint columns
    # Qui uso df_test = scale_joint_columns(df_test, use_existing_scaler=True)
    # Questa è la pratica corretta per evitare data leakage
    df = scale_joint_columns(df)
    df_test = scale_joint_columns(df_test, use_existing_scaler=True)
    
    # Prepare data for embeddings (ensures categorical columns are int64)
    df, df_test, categorical_cols, continuous_cols = prepare_data_with_embeddings(df, df_test)
    
    # Apply target weighting
    target = apply_target_weighting(target)
    
    # Train/val split
    train_df, val_df, train_target, val_target = train_val_split(df, target, val_ratio=0.2)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Train samples: {len(train_df['sample_index'].unique())}")
    print(f"Val samples: {len(val_df['sample_index'].unique())}")
    print(f"Test samples: {len(df_test['sample_index'].unique())}")
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Continuous columns ({len(continuous_cols)}): {continuous_cols[:5]}... (+{len(continuous_cols)-5} more)")
    print("="*70)
    
    return train_df, val_df, train_target, val_target, df_test, categorical_cols, continuous_cols


def run_test_preprocessing():
    """
    DEPRECATED: Use run_preprocessing() instead, which returns test data.
    
    This function is kept for backward compatibility but is no longer needed.
    The main run_preprocessing() now returns test data along with train/val.
    """
    print("WARNING: run_test_preprocessing() is deprecated.")
    print("Use run_preprocessing() instead, which returns test data.")
    
    df_test = pd.read_csv("pirate_pain_test.csv")
    df_test = df_test.drop(columns=['joint_30'])
    df_test = df_test.drop(columns=['joint_11'])
    df_test = df_test.drop(columns=['time'])
    
    # Add time-based features (November 12 clue implementation)
    df_test, _ = add_time_features(df_test, df_test)
    df_test, _ = add_prosthetics_feature(df_test, df_test)
    df_test = scale_joint_columns(df_test, use_existing_scaler=True)
    
    # Prepare for embeddings
    df_test, _, _, _ = prepare_data_with_embeddings(df_test, df_test)

    return df_test