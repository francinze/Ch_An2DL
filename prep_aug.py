## Augmentation
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import autoaugment, transforms, functional
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def augment_images(
    train_img_dir: str, 
    train_mask_dir: str, 
    train_labels: pd.DataFrame, 
    augmentation_policy: str,
    target_samples: int = 1000
):
    """
    Augment images based on class distribution and specified augmentation policy.
    Saves augmented images to the same directories as originals.
    Parameters:
    - augmentation_policy: 'autoaugment', 'randaugment', or 'trivialaugment'
    """
    # Remove any previously augmented images from the training set
    train_labels = train_labels[~train_labels['sample_index'].str.contains('_aug_')].reset_index(drop=True)
    for img_file in os.listdir(train_img_dir):
        if '_aug_' in img_file:
            os.remove(os.path.join(train_img_dir, img_file))
    for mask_file in os.listdir(train_mask_dir):
        if '_aug_' in mask_file:
            os.remove(os.path.join(train_mask_dir, mask_file))


    # Analyze class distribution after removal
    class_distribution = train_labels['label'].value_counts().sort_index()
    print("\n" + "="*60)
    print("Class Distribution After Removal of Contaminated Images")
    print("="*60)
    print(class_distribution)
    print(f"\nTotal samples: {len(train_labels)}")

    # Calculate statistics
    print("\n" + "="*60)
    print("STATISTICS FOR AUGMENTATION")
    print("="*60)

    # Class with the most samples (majority)
    max_class = class_distribution.max()
    max_class_name = class_distribution.idxmax()
    print(f"\nClass with the most samples (Majority): {max_class_name} ({max_class} samples)")

    # Class with the fewest samples (minority)
    min_class = class_distribution.min()
    min_class_name = class_distribution.idxmin()
    print(f"Class with the fewest samples (Minority): {min_class_name} ({min_class} samples)")

    # Imbalance ratio
    imbalance_ratio = max_class / min_class
    print(f"\nImbalance ratio (Max/Min): {imbalance_ratio:.2f}x")

    # Augmentation proposal
    print("\n" + "="*60)
    print("RECOMMENDED AUGMENTATION STRATEGY")
    print("="*60)
    print("\nAugmentations to apply (as suggested by the professor):")
    print("  1. Horizontal Flip (p=0.5)")
    print("  2. Vertical Flip (p=0.5)")
    print("  3. Random Translation (0.2, 0.2)")
    print("  4. Random Zoom/Scale (0.8, 1.2)")
    print("  [EXCLUDE: Random Rotation - would change dimensions]\n")

    # STRATEGY: All classes grow until reaching the same target number for ALL
    print("\n" + "="*80)
    print("BALANCED STRATEGY: ALL CLASSES GROW TO A FIXED AND EQUAL NUMBER")
    print("="*80)

    print(f"\nTarget: {target_samples} samples for EACH class")

    augmentation_strategy_balanced = {}
    total_to_generate = 0

    for class_name in class_distribution.index:
        n_samples = class_distribution[class_name]
        n_needed = target_samples - n_samples
        n_augmentations = max(0, n_needed)  # We cannot have negative augmentations
        
        augmentation_strategy_balanced[class_name] = {
            'original': n_samples,
            'target': target_samples,
            'augment_count': n_augmentations,
            'ratio_multiplier': n_augmentations / n_samples if n_samples > 0 else 0
        }
        
        total_to_generate += n_augmentations

    # Projection of the dataset after augmentation
    print("\n" + "="*80)
    print("DATASET AFTER BALANCED AUGMENTATION")
    print("="*80)
    print(f"{'Class':<20} {'Original':<15} {'New Augment':<15} {'Augmentations per image':<25} {'Total':<15}")
    print("-" * 80)

    total_original = 0
    total_augmented = 0
    for class_name in class_distribution.index:
        n_original = class_distribution[class_name]
        n_aug = augmentation_strategy_balanced[class_name]['augment_count']
        n_total = n_original + n_aug
        
        total_original += n_original
        total_augmented += n_total
        
        print(f"{class_name:<20} {n_original:<15} {n_aug:<15} {augmentation_strategy_balanced[class_name]['ratio_multiplier']:<25.2f} {n_total:<15}")

    print("-" * 80)
    print(f"{'TOTAL':<20} {total_original:<15} {total_to_generate:<15} {np.mean([augmentation_strategy_balanced[class_name]['ratio_multiplier'] for class_name in class_distribution.index]):<25.2f} {total_augmented:<15}")

    # Visualize the distribution before and after
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before
    class_distribution.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Class Distribution - BEFORE Augmentation', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of samples')
    axes[0].set_xlabel('Class')
    axes[0].axhline(y=target_samples, color='red', linestyle='--', linewidth=2, label=f'Target: {target_samples}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # After
    after_augmentation_balanced = {}
    for class_name in class_distribution.index:
        after_augmentation_balanced[class_name] = augmentation_strategy_balanced[class_name]['target']

    after_series = pd.Series(after_augmentation_balanced)
    after_series.plot(kind='bar', ax=axes[1], color='seagreen')
    axes[1].set_title('Class Distribution - AFTER Balanced Augmentation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of samples')
    axes[1].set_xlabel('Class')
    axes[1].axhline(y=target_samples, color='red', linestyle='--', linewidth=2, label=f'Target: {target_samples}')
    axes[1].set_ylim([0, max_class * 1.1])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Define Automatic Augmentation Strategy
    print("\n" + "="*80)
    print("AUTOMATIC AUGMENTATION CONFIGURATION")
    print("="*80)

    # Choose augmentation policy
    img_auto_augment = None
    if augmentation_policy == 'autoaugment':
        auto_policy = autoaugment.AutoAugmentPolicy.IMAGENET
        img_auto_augment = autoaugment.AutoAugment(policy=auto_policy)
        print(f"Using AutoAugment with policy: {auto_policy}")
        print("  - Automatically applies learned augmentation combinations")
        print("  - Optimized for ImageNet-like datasets")
        
    elif augmentation_policy == 'randaugment':
        num_ops = 2
        magnitude = 9
        img_auto_augment = autoaugment.RandAugment(num_ops=num_ops, magnitude=magnitude)
        print(f"Using RandAugment:")
        print(f"  - num_ops: {num_ops} (number of augmentations per image)")
        print(f"  - magnitude: {magnitude} (strength 0-30)")
        print("  - Randomly selects and applies augmentation operations")
        
    elif augmentation_policy == 'trivialaugment':
        img_auto_augment = autoaugment.TrivialAugmentWide()
        print("Using TrivialAugmentWide:")
        print("  - Applies one random augmentation per image")
        print("  - No hyperparameters needed")
        print("  - Often competitive with more complex strategies")

    print("\nMask augmentation:")
    print("  - Synchronized geometric transforms only (flip operations)")
    print("  - No color/brightness changes (masks are binary/grayscale)")

    # IMPORTANT: We apply AUTOMATIC augmentation to images
    # and SYNCHRONIZED GEOMETRIC transforms to masks
    print("\n" + "="*80)
    print("STARTING DUAL-PATH AUTOMATIC AUGMENTATION PROCESS")
    print(f"Augmentation Policy: {augmentation_policy}")
    print("Augmenting both IMAGES and MASKS")
    print("  - Images: Automatic augmentation (color, geometric, etc.)")
    print("  - Masks: Synchronized geometric transforms only")
    print("Saving augmented files to same directories as originals")
    print("="*80)

    # Track augmented files for labels
    augmented_files = []
    total_augmented = 0

    for class_name in sorted(augmentation_strategy_balanced.keys()):
        info = augmentation_strategy_balanced[class_name]
        n_augment = info['augment_count']
        
        if n_augment == 0:
            print(f"\n{class_name}: No augmentation needed (already at target)")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Class: {class_name}")
        print(f"Augmentations to generate: {n_augment}")
        print(f"{'-'*80}")
        
        # Get ALL images of this class (real + synthetic)
        class_samples = train_labels[train_labels['label'] == class_name]['sample_index'].tolist()
        n_total = len(class_samples)
        
        # Calculate how many augmentations per image (real + synthetic)
        aug_per_img = n_augment / n_total
        
        # For each image (real or synthetic)
        aug_count = 0
        for img_idx, img_name in enumerate(tqdm(class_samples, desc=f"  {class_name}")):
            # Load BOTH the image AND mask
            img_file = img_name
            mask_file = img_name.replace('img_', 'mask_')
            
            img_path = os.path.join(train_img_dir, img_file)
            mask_path = os.path.join(train_mask_dir, mask_file)
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"  File not found: {img_file} or {mask_file}")
                continue
            
            # Load image and mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale mask
            
            # Generate augmentations for this image-mask pair
            n_to_generate = int(np.ceil(aug_per_img)) if img_idx < n_augment % n_total else int(np.floor(aug_per_img))
            
            for aug_num in range(n_to_generate):
                if aug_count < n_augment:
                    base_name = img_file.replace('.png', '')
                    mask_base_name = mask_file.replace('.png', '')
                    
                    # Apply AUTOMATIC augmentation to image
                    img_augmented = img_auto_augment(img)
                    
                    # Apply synchronized GEOMETRIC transforms to mask
                    seed = np.random.randint(2147483647)
                    
                    # Horizontal flip
                    torch.manual_seed(seed)
                    mask_augmented = transforms.RandomHorizontalFlip(p=0.5)(mask)
                    
                    # Vertical flip  
                    torch.manual_seed(seed)
                    mask_augmented = transforms.RandomVerticalFlip(p=0.5)(mask_augmented)
                    
                    # Random 90-degree rotations
                    rotation_angle = np.random.choice([0, 90, 180, 270])
                    mask_augmented = functional.rotate(
                        mask_augmented, 
                        angle=int(rotation_angle), 
                        expand=True, 
                        fill=[0]
                    )
                    
                    # Apply same rotation to image to maintain consistency
                    img_augmented = functional.rotate(
                        img_augmented,
                        angle=int(rotation_angle),
                        expand=True,
                        fill=[0]
                    )
                    
                    # Save augmented image and mask to same directories
                    augmented_img_name = f"{base_name}_aug_{aug_count}.png"
                    augmented_mask_name = f"{mask_base_name}_aug_{aug_count}.png"
                    
                    augmented_img_path = os.path.join(train_img_dir, augmented_img_name)
                    augmented_mask_path = os.path.join(train_mask_dir, augmented_mask_name)
                    
                    # Convert tensors to PIL Images if necessary
                    if torch.is_tensor(img_augmented):
                        img_augmented = transforms.ToPILImage()(img_augmented)
                    if torch.is_tensor(mask_augmented):
                        mask_augmented = transforms.ToPILImage()(mask_augmented)
                    
                    img_augmented.save(augmented_img_path)
                    mask_augmented.save(augmented_mask_path)
                    
                    # Track for labels
                    augmented_files.append({'sample_index': augmented_img_name, 'label': class_name})
                    
                aug_count += 1
        
        total_augmented += aug_count
        print(f"  {class_name}: Completed! {aug_count} image-mask pairs generated")

    print("\n" + "="*80)
    print(f"AUTOMATIC AUGMENTATION COMPLETED!")
    print(f"Total augmented image-mask pairs generated: {total_augmented}")
    print(f"Augmented images added to: {train_img_dir}")
    print(f"Augmented masks added to: {train_mask_dir}")
    print("="*80)

    # Verify file count
    img_files = os.listdir(train_img_dir)
    mask_files = os.listdir(train_mask_dir)
    print(f"\nTotal images in directory: {len(img_files)}")
    print(f"Total masks in directory: {len(mask_files)}")

    # Select a random sample from the original dataset for visualization
    sample_idx = np.random.randint(0, len(train_labels))
    sample_row = train_labels.iloc[sample_idx]
    sample_name = sample_row['sample_index']
    sample_label = sample_row['label']

    img_path = os.path.join(train_img_dir, sample_name)
    mask_path = os.path.join(train_mask_dir, sample_name.replace('img_', 'mask_'))

    if os.path.exists(img_path) and os.path.exists(mask_path):
        original_img = Image.open(img_path).convert('RGB')
        original_mask = Image.open(mask_path).convert('L')
        
        # Generate multiple augmented versions for visualization
        n_examples = 8
        fig, axes = plt.subplots(3, n_examples, figsize=(20, 8))
        
        # Show original and augmented examples
        for i in range(n_examples):
            if i == 0:
                axes[0, i].imshow(original_img)
                axes[0, i].set_title(f"Original\n{sample_label}", fontsize=10)
                
                axes[1, i].imshow(original_mask, cmap='gray')
                axes[1, i].set_title("Original Mask", fontsize=10)
                
                # Composite - resize mask to match image dimensions
                mask_resized = original_mask.resize(original_img.size, Image.NEAREST)
                composite = Image.blend(original_img.convert('RGB'), 
                                    mask_resized.convert('RGB'), 
                                    alpha=0.3)
                axes[2, i].imshow(composite)
                axes[2, i].set_title("Composite", fontsize=10)
            else:
                # Generate augmented version for visualization
                aug_img = img_auto_augment(original_img)
                
                # Apply matching geometric transform to mask
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                aug_mask = transforms.RandomHorizontalFlip(p=0.5)(original_mask)
                torch.manual_seed(seed)
                aug_mask = transforms.RandomVerticalFlip(p=0.5)(aug_mask)
                
                rotation = np.random.choice([0, 90, 180, 270])
                aug_mask = functional.rotate(aug_mask, angle=int(rotation), expand=True, fill=[0])
                aug_img = functional.rotate(aug_img, angle=int(rotation), expand=True, fill=[0])
                
                # Convert tensors to PIL Images if necessary
                if torch.is_tensor(aug_img):
                    aug_img = transforms.ToPILImage()(aug_img)
                if torch.is_tensor(aug_mask):
                    aug_mask = transforms.ToPILImage()(aug_mask)
                
                axes[0, i].imshow(aug_img)
                axes[0, i].set_title(f"Aug #{i}\n{augmentation_policy}", fontsize=10)
                
                axes[1, i].imshow(aug_mask, cmap='gray')
                axes[1, i].set_title(f"Aug Mask #{i}", fontsize=10)
                
                # Composite
                if aug_img.size == aug_mask.size:
                    composite = Image.blend(aug_img.convert('RGB'),
                                        aug_mask.convert('RGB'),
                                        alpha=0.3)
                    axes[2, i].imshow(composite)
                    axes[2, i].set_title(f"Composite #{i}", fontsize=10)
            
            for ax in axes[:, i]:
                ax.axis('off')
        
        plt.suptitle(f"Automatic Augmentation Examples - {augmentation_policy.upper()}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"\n✓ Showing {n_examples} augmentation variations")
        print(f"  Policy: {augmentation_policy}")
        print(f"  Notice the variety in color, brightness, contrast, and geometric transforms")
    else:
        print("Sample images not found for visualization")

    # Create augmented dataframe and combine with original
    augmented_df = pd.DataFrame(augmented_files)
    train_labels_augmented = pd.concat([train_labels, augmented_df], ignore_index=True)

    return train_labels_augmented
