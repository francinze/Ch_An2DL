from typing import Any
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def load_patches(
    train_img_dir: str, test_img_dir: str, train_labels: pd.DataFrame, path_prefix: str, 
    batch_size: int, patch_size: int, patch_stride: int, images_per_batch: int
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if torch.cuda.is_available():
        num_workers = 2 * torch.cuda.device_count()  # 2 workers per GPU = 4 total for T4 x2
        pin_memory = True  # CRITICAL: Enables async CPU-to-GPU transfer while GPU computes
        persistent_workers = True  # Keeps workers alive between epochs (faster)
        print(f"GPU detected: {torch.cuda.device_count()} GPU(s) - using {num_workers} workers")
    else:
        # CPU-only environment
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    # ======================================

    # Load original + augmented images into tensors
    print("\n" + "="*80)
    print("LOADING IMAGES")
    print("="*80)

    print(f"\nDistribution in dataset:")
    print(train_labels['label'].value_counts().sort_index())

    # ===== PRE-EXTRACT PATCHES TO DISK (RUN ONCE) =====
    print("\n" + "="*80)
    print("PATCH PRE-EXTRACTION (Multi-File Strategy)")
    print("="*80)

    # Directory to save pre-extracted patches
    patches_dir = os.path.join(path_prefix, "data", "patches_cache")
    os.makedirs(patches_dir, exist_ok=True)

    # Metadata file to track all batch files
    metadata_file = os.path.join(patches_dir, f"metadata_ps{patch_size}_stride{patch_stride}.json")

    if os.path.exists(metadata_file):
        print(f"✓ Found pre-extracted patches!")
        print(f"  Loading metadata from: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Found {metadata['num_batches']} batch files with {metadata['total_patches']} total patches")
        print(f"✓ Found {metadata['num_batches']} batch files with {metadata['total_patches']} total patches")
        
    else:
        print(f"⚠ No pre-extracted patches found. Extracting now...")
        print(f"  Strategy: Save each batch to separate file (no huge concatenation!)")
        
        # Helper function to extract patches from a single image
        def extract_patches_from_image(img_path: str, patch_size: int, stride: int, min_variance_threshold: float = 0.001) -> list[torch.Tensor]:
            """
            Extract all patches from a single image, filtering out blank patches.
            
            Args:
                img_path: Path to the image file
                patch_size: Size of each patch
                stride: Stride for patch extraction
                min_variance_threshold: Minimum variance to consider patch non-blank (default: 0.001)
            
            Returns:
                List of patch tensors (excluding blank patches)
            """
            img = Image.open(img_path).convert('RGB')
            
            img_array = np.array(img, dtype=np.float32)
            h, w, _ = img_array.shape
            patches: list[torch.Tensor] = []
            
            if h < patch_size or w < patch_size:
                pad_h = max(0, patch_size - h)
                pad_w = max(0, patch_size - w)
                img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                patch = img_array[:patch_size, :patch_size, :]
                
                # Check if patch is not blank
                patch_normalized = patch / 255.0
                if np.var(patch_normalized) > min_variance_threshold:
                    patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1) / 255.0).float()
                    patches.append(patch_tensor)
            else:
                n_patches_h = max(1, (h - patch_size) // stride + 1)
                n_patches_w = max(1, (w - patch_size) // stride + 1)
                
                for row_idx in range(n_patches_h):
                    for col_idx in range(n_patches_w):
                        start_h = min(row_idx * stride, h - patch_size)
                        start_w = min(col_idx * stride, w - patch_size)
                        patch = img_array[start_h:start_h+patch_size, start_w:start_w+patch_size, :]
                        
                        # Check if patch is not blank (has sufficient variance)
                        patch_normalized = patch / 255.0
                        if np.var(patch_normalized) > min_variance_threshold:
                            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1) / 255.0).float()
                            patches.append(patch_tensor)
            
            return patches
        
        label_map = {'Triple negative': 0, 'Luminal A': 1, 'Luminal B': 2, 'HER2(+)': 3}
        
        print(f"\nExtracting and saving patches in batches of {images_per_batch} images...")
        
        batch_patches: list[torch.Tensor] = []
        batch_labels: list[int] = []
        batch_num = 0
        total_patches = 0
        batch_files: list[str] = []
        
        for idx, row in train_labels.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(train_labels)} images...")
            
            img_name = row['sample_index']
            label = label_map[row['label']]
            
            img_path = os.path.join(train_img_dir, img_name)
            
            if os.path.exists(img_path):
                patches = extract_patches_from_image(img_path, patch_size, patch_stride)
                batch_patches.extend(patches)
                batch_labels.extend([label] * len(patches))
            
            # Save batch to disk when full (DON'T accumulate in memory!)
            if (idx + 1) % images_per_batch == 0 or idx == len(train_labels) - 1:
                if len(batch_patches) > 0:
                    batch_tensor = torch.stack(batch_patches)
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
                    
                    # Save this batch to its own file
                    batch_file = os.path.join(patches_dir, f"batch_{batch_num:03d}.pt")
                    torch.save({'patches': batch_tensor, 'labels': batch_labels_tensor}, batch_file)
                    
                    batch_files.append(batch_file)
                    total_patches += len(batch_patches)
                    print(f"    ✓ Saved batch {batch_num}: {len(batch_patches)} patches → {batch_file}")
                    
                    # CRITICAL: Clear memory immediately!
                    batch_patches = []
                    batch_labels = []
                    batch_num += 1
        
        # Save metadata
        metadata: dict[str, Any] = {
            'num_batches': batch_num,
            'total_patches': total_patches,
            'batch_files': batch_files,
            'patch_size': patch_size,
            'stride': patch_stride
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"  Saved {batch_num} batch files with {total_patches} total patches")
        print(f"  Metadata saved to: {metadata_file}")

    # Load metadata for dataset creation
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"\n{'='*80}")
    print("Creating multi-file dataset...")
    print(f"Total patches: {metadata['total_patches']}")

    # Custom Dataset that loads from multiple batch files
    class MultiFilePatchDataset(Dataset):
        """Dataset that loads patches from multiple batch files with caching"""
        
        def __init__(self, metadata: dict[str, Any]):
            self.batch_files: list[str] = metadata['batch_files']
            self.total_patches = metadata['total_patches']
            
            # Cache for loaded batches (key: batch_idx, value: batch_data)
            self._cache: dict[int, dict[str, Any]] = {}
            
            # Load all batches to build index (loads only metadata, not actual tensors)
            self.batch_sizes: list[int] = []
            self.cumulative_sizes: list[int] = [0]
            
            print(f"Loading batch metadata from {len(self.batch_files)} files...")
            for batch_file in self.batch_files:
                batch_data = torch.load(batch_file)
                batch_size = len(batch_data['patches'])
                self.batch_sizes.append(batch_size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + batch_size)
                
                # CRITICAL: Pre-load all batches into cache (only ~22K patches total, manageable)
                # This avoids repeated torch.load() calls which are extremely slow
                self._cache[len(self.batch_sizes) - 1] = batch_data
            
            print(f"Dataset ready with {self.total_patches} patches (all batches cached in RAM)")
        
        def __len__(self):
            return self.total_patches
        
        def __getitem__(self, idx: int):
            # Find which batch file contains this index
            batch_idx: int = int(np.searchsorted(self.cumulative_sizes[1:], idx, side='right'))
            local_idx = idx - self.cumulative_sizes[batch_idx]
            
            # Use cached batch data (no disk I/O!)
            batch_data = self._cache[batch_idx]
            
            return batch_data['patches'][local_idx], batch_data['labels'][local_idx]

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"\n{'='*80}")
    print("Creating multi-file dataset...")
    print(f"Total patches: {metadata['total_patches']}")

    # Create dataset
    full_dataset = MultiFilePatchDataset(metadata)

    # ===== CREATE TRAIN/VAL SPLIT =====
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*80)

    # Create indices for train/val split
    # Need to load all labels to stratify properly
    print("Loading all labels for stratified split...")
    all_labels_list: list[Any] = []
    for batch_file in metadata['batch_files']:
        batch_data = torch.load(batch_file)
        all_labels_list.extend(batch_data['labels'].tolist())

    all_labels_array: np.ndarray[Any, Any] = np.array(all_labels_list)

    # Split indices (not data!)
    train_indices, val_indices = train_test_split(
        np.arange(len(all_labels_array)), test_size=0.2, random_state=42, 
        stratify=all_labels_array
    )

    print(f"Train patches: {len(train_indices)}")
    print(f"Val patches: {len(val_indices)}")

    # Create subset datasets
    train_dataset: Subset[Any] = Subset(full_dataset, train_indices)
    val_dataset: Subset[Any] = Subset(full_dataset, val_indices)

    label_map = {'Triple negative': 0, 'Luminal A': 1, 'Luminal B': 2, 'HER2(+)': 3}

    # Create DataLoaders with GPU optimizations
    train_loader_kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    val_loader_kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = persistent_workers
        val_loader_kwargs['persistent_workers'] = persistent_workers

    train_loader: DataLoader[Any] = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader: DataLoader[Any] = DataLoader(val_dataset, **val_loader_kwargs)

    print(f"\nOptimization: {num_workers} workers, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
    print(f"\nCreated DataLoaders:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("="*80)

    # Memory-efficient test dataset
    class TestPatchDataset(Dataset):
        """Memory-efficient test dataset that extracts patches on-the-fly"""
        
        def __init__(self):
            self.patch_size = patch_size
            
            # Get image files
            self.image_files = sorted([f for f in os.listdir(test_img_dir) if f.startswith('img_')])
            
            # Pre-compute patch counts and mapping
            self.patch_counts: list[int] = []
            self.cumulative_patches: list[int] = [0]
            self.patch_to_image: list[int] = []
            self.stride = patch_stride
            
            print(f"Computing patch counts for {len(self.image_files)} test images...")
            for i, img_name in enumerate(self.image_files):
                if i % 100 == 0 and i > 0:
                    print(f"  Processed {i}/{len(self.image_files)} images...")
                
                img_path = os.path.join(test_img_dir, img_name)
                img = Image.open(img_path)
                h, w = img.size[1], img.size[0]
                
                # Calculate patches
                n_patches_h = max(1, (h - patch_size) // self.stride + 1)
                n_patches_w = max(1, (w - patch_size) // self.stride + 1)
                n_patches = n_patches_h * n_patches_w
                
                if h < patch_size or w < patch_size:
                    n_patches = 1
                
                self.patch_counts.append(n_patches)
                self.cumulative_patches.append(self.cumulative_patches[-1] + n_patches)
                
                # Track which image each patch belongs to
                for _ in range(n_patches):
                    self.patch_to_image.append(i)
            
            self.total_patches = self.cumulative_patches[-1]
            print(f"Total test patches: {self.total_patches}")
        
        def __len__(self):
            return self.total_patches
        
        def _extract_patch(self, image: np.ndarray, patch_idx: int, img_h: int, img_w: int):
            """Extract a specific patch from an image"""
            n_patches_h = max(1, (img_h - self.patch_size) // self.stride + 1)
            n_patches_w = max(1, (img_w - self.patch_size) // self.stride + 1)
            
            row_idx = patch_idx // n_patches_w
            col_idx = patch_idx % n_patches_w
            
            start_h = min(row_idx * self.stride, img_h - self.patch_size)
            start_w = min(col_idx * self.stride, img_w - self.patch_size)
            
            patch: np.ndarray = image[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
            return patch
        
        def __getitem__(self, idx: int):
            # Find image for this patch
            img_idx = np.searchsorted(self.cumulative_patches[1:], idx, side='right')
            patch_idx = idx - self.cumulative_patches[img_idx]
            img_name = self.image_files[img_idx]
            
            img_path = os.path.join(test_img_dir, img_name)
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Extract patch
            h, w, c = img_array.shape
            
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                patch = img_array[:self.patch_size, :self.patch_size, :]
            else:
                patch: np.ndarray = self._extract_patch(img_array, patch_idx, h, w)
            
            final_img: np.ndarray = patch
        
            # Convert to tensor
            img_tensor = torch.from_numpy(final_img).permute(2, 0, 1).float() / 255.0
            
            return img_tensor

    # Load test data
    print(f"Using PATCH-BASED processing for test data")
        
    test_dataset = TestPatchDataset()
    test_filenames = test_dataset.image_files

    print(f"Test patches: {len(test_dataset)}")
    print(f"Test image files: {len(test_filenames)}")

    # Create DataLoader with GPU optimizations
    test_loader_kwargs: dict[str, object] = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if num_workers > 0:
        test_loader_kwargs['persistent_workers'] = persistent_workers
    test_loader: DataLoader = DataLoader(test_dataset, **test_loader_kwargs)

    print(f"\nDataLoader created:")
    print(f"Test batches: {len(test_loader)}")
    print(f"Optimization: {num_workers} workers, pin_memory={pin_memory}")

    input_shape = (3, patch_size, patch_size)
    num_classes = len(label_map)

    return train_loader, val_loader, test_loader, input_shape, num_classes