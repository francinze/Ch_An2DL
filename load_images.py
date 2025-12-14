import os
from torch.utils.data import TensorDataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

def load_images(train_dir: str, test_dir: str, train_labels: pd.DataFrame, img_size=(224, 224), batch_size=32):
    if torch.cuda.is_available():
        num_workers = min(8, 4 * torch.cuda.device_count())  # 4 workers per GPU
        pin_memory = True  # Faster CPU-to-GPU transfer
        persistent_workers = True if num_workers > 0 else False
    else:
        # CPU-only environment
        num_workers = 0  # Avoid multiprocessing overhead on CPU
        pin_memory = False
        persistent_workers = False
    # ======================================

    # Load original + augmented images into tensors
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    print(f"Augmented directory: {train_dir}")

    print(f"\nDistribution in dataset:")
    print(train_labels['label'].value_counts().sort_index())

    print("\nLoading images into tensors...")
    images = []
    labels = []
    
    for _, row in train_labels.iterrows():
        img_name = row['sample_index']
        label = row['label']

        img_path = os.path.join(train_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Image not found: {img_path}")
            continue
        try:
            # Load as RGB
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size, Image.BILINEAR)
            img_array = np.array(img)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            continue
        images.append(img_array)
        labels.append(label)
    
    # Convert to tensors
    images = np.array(images)
    X_train = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    
    label_map = {'Triple negative': 0, 'Luminal A': 1, 'Luminal B': 2, 'HER2(+)': 3}
    label_indices = [label_map[label] for label in labels]
    y_train = torch.tensor(label_indices, dtype=torch.long)

    print(f"Images tensor shape: {X_train.shape}")
    print(f"Labels tensor shape: {y_train.shape}")

    # Split training/validation (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain set augmented: {X_train.shape[0]} samples")
    print(f"Validation set augmented: {X_val.shape[0]} samples")

    # Create new DataLoaders with GPU optimizations
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # DataLoader configuration (conditional persistent_workers)
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    val_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    # Only add persistent_workers if num_workers > 0 (not supported otherwise)
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = persistent_workers
        val_loader_kwargs['persistent_workers'] = persistent_workers

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    print(f"Optimization: {num_workers} workers, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

    print(f"\nCreated DataLoaders:")
    print(f"Val batches: {len(val_loader)}")
    print(f"Train batches: {len(train_loader)}")

    # Load test data
    print(f"\nLoading test data")
    image_files = sorted([f for f in os.listdir(test_dir) if f.startswith('img_')])
    
    images = []
    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize(img_size, Image.BILINEAR)
        img_array = np.array(img)
        
        images.append(img_array)
    
    # Stack into numpy array: (N, H, W, C)
    images = np.array(images)
    # Convert to tensor and permute to (N, C, H, W)
    X_test = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    print(f"Test images shape: {X_test.shape}")

    test_dataset = TensorDataset(X_test)

    # Create DataLoader with GPU optimizations
    test_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if num_workers > 0:
        test_loader_kwargs['persistent_workers'] = persistent_workers

    test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    print(f"\nDataLoader created:")
    print(f"Test batches: {len(test_loader)}")
    print(f"Optimization: {num_workers} workers, pin_memory={pin_memory}")

    input_shape = X_train.shape[1:]  # (C, H, W)
    num_classes = len(label_map)

    return train_loader, val_loader, test_loader, input_shape, num_classes