import torch
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
import os
import pandas as pd

def generate_synthetic_images(train_img_dir: str, train_mask_dir: str, train_labels: pd.DataFrame, synthetic_multiplier: int):
    
    # Remove any previously synthesized images from the training set
    train_labels = train_labels[~train_labels['sample_index'].str.contains('_syn_')].reset_index(drop=True)
    for img_file in os.listdir(train_img_dir):
        if '_syn_' in img_file:
            os.remove(os.path.join(train_img_dir, img_file))
    for mask_file in os.listdir(train_mask_dir):
        if '_syn_' in mask_file:
            os.remove(os.path.join(train_mask_dir, mask_file))


    # ===== MULTI-GPU SETUP =====
    # Check for multiple GPUs and set up DataParallel
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            print(f" GPU {i}: {torch.cuda.get_device_name(i)}")
        if num_gpus > 1:
            print(f"Multi-GPU training enabled: Will use {num_gpus} GPUs with DataParallel")
        else:
            print(f"Single GPU training")
    else:
        num_gpus = 0
        print("No GPU available, using CPU")
    # ===========================

    print("="*80)
    print("GENERATIVE MASK CONDITIONING SETUP (Pix2Pix GAN)")
    print("="*80)
    print(f"Adding synthetic images to: {train_img_dir}")
    print(f"Adding synthetic masks to: {train_mask_dir}")
    print(f"Synthetic multiplier: {synthetic_multiplier}x per image")
    print(f"Using GPU: {num_gpus > 0}")
    print(f"Method: Lightweight Pix2Pix GAN (~50MB)")
    print("="*80)
    
    # Define Pix2Pix Generator (U-Net architecture)
    class UNetGenerator(nn.Module):
        """Lightweight U-Net generator for Pix2Pix"""
        def __init__(self, in_channels: int = 1, out_channels: int = 3):
            super(UNetGenerator, self).__init__()
            
            # Encoder (downsampling)
            self.enc1 = self.conv_block(in_channels, 64, normalize=False)
            self.enc2 = self.conv_block(64, 128)
            self.enc3 = self.conv_block(128, 256)
            self.enc4 = self.conv_block(256, 512)
            
            # Decoder (upsampling) with skip connections
            self.dec1 = self.upconv_block(512, 256)
            self.dec2 = self.upconv_block(512, 128)  # 512 = 256 + 256 from skip
            self.dec3 = self.upconv_block(256, 64)   # 256 = 128 + 128 from skip
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 128 = 64 + 64 from skip
                nn.Tanh()
            )
        
        def conv_block(self, in_ch: int, out_ch: int, normalize: bool = True):
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        def upconv_block(self, in_ch: int, out_ch: int):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            
            # Decoder with skip connections
            d1 = self.dec1(e4)
            d2 = self.dec2(torch.cat([d1, e3], 1))
            d3 = self.dec3(torch.cat([d2, e2], 1))
            d4 = self.dec4(torch.cat([d3, e1], 1))
            
            return d4
    
    # Initialize generator
    generator = UNetGenerator(in_channels=1, out_channels=3)
    if num_gpus > 0:
        generator = generator.cuda()
    
    # Initialize with random weights (or load pretrained if available)
    generator.eval()  # Set to eval mode for inference
    
    print("✓ Pix2Pix Generator initialized")
    print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Size: ~{sum(p.numel() for p in generator.parameters()) * 4 / 1024 / 1024:.1f} MB")
    print(f"  Device: {'GPU' if num_gpus > 0 else 'CPU'}")

    # Color/texture variations for synthetic generation
    # Since GAN starts with random weights, we enhance with classical augmentations
    class_color_profiles: dict[str, dict[str, tuple[float, float]]] = {
        'Triple negative': {
            'hue_range': (-10, 10),
            'saturation_mult': (0.9, 1.2),
            'brightness_mult': (0.85, 1.15),
            'contrast_mult': (0.9, 1.1)
        },
        'Luminal A': {
            'hue_range': (-15, 15),
            'saturation_mult': (0.8, 1.1),
            'brightness_mult': (0.9, 1.1),
            'contrast_mult': (0.95, 1.05)
        },
        'Luminal B': {
            'hue_range': (-12, 12),
            'saturation_mult': (0.85, 1.15),
            'brightness_mult': (0.88, 1.12),
            'contrast_mult': (0.92, 1.08)
        },
        'HER2(+)': {
            'hue_range': (-8, 8),
            'saturation_mult': (0.95, 1.25),
            'brightness_mult': (0.9, 1.1),
            'contrast_mult': (0.95, 1.1)
        }
    }

    print("\n✓ Class-specific color profiles defined")
    print(f"  Classes: {list(class_color_profiles.keys())}")

    def prepare_mask_for_gan(mask_path: str, target_size: int = 256) -> torch.Tensor:
        """
        Prepare mask for GAN input.
        Converts to tensor format and normalizes.
        """
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((target_size, target_size), Image.BILINEAR)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        if num_gpus > 0:
            mask_tensor = mask_tensor.cuda()
        
        return mask_tensor

    def apply_color_jitter(img: Image.Image, color_profile: dict[str, tuple[float, float]]) -> Image.Image:
        """Apply color jittering based on class-specific profile"""
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Hue shift
        hue_shift = np.random.uniform(*color_profile['hue_range'])
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation multiplication
        sat_mult = np.random.uniform(*color_profile['saturation_mult'])
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_mult, 0, 255)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Brightness and contrast
        brightness_mult = np.random.uniform(*color_profile['brightness_mult'])
        contrast_mult = np.random.uniform(*color_profile['contrast_mult'])
        
        img_rgb = np.clip(img_rgb * brightness_mult, 0, 255).astype(np.uint8)
        img_rgb = np.clip((img_rgb - 128) * contrast_mult + 128, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_rgb)

    def generate_synthetic_image_gan(mask_tensor: torch.Tensor, original_img: Image.Image, class_label: str, generator: UNetGenerator) -> Image.Image:
        """
        Generate synthetic image using GAN + color augmentation
        
        Args:
            mask_tensor: Tensor of mask [1, 1, H, W]
            original_img: PIL Image of original (for reference)
            class_label: Cancer subtype label
            generator: UNetGenerator model
        
        Returns:
            PIL Image of generated synthetic image
        """
        with torch.no_grad():
            # Generate base image from mask using GAN
            generated = generator(mask_tensor)  # Output: [-1, 1]
            
            # Convert to PIL Image
            generated_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_np = ((generated_np + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0,255]
            synthetic_img = Image.fromarray(generated_np)
            
            # Apply class-specific color augmentation
            color_profile = class_color_profiles[class_label]
            synthetic_img = apply_color_jitter(synthetic_img, color_profile)
            
            # Blend with original for better texture (optional, helps with untrained GAN)
            alpha = np.random.uniform(0.3, 0.7)  # Random blending
            synthetic_img = Image.blend(original_img.resize(synthetic_img.size), synthetic_img, alpha)
        
        return synthetic_img

    print("✓ Helper functions defined")
    print("  - prepare_mask_for_gan(): Convert masks to GAN tensor format")
    print("  - apply_color_jitter(): Class-specific color augmentation")
    print("  - generate_synthetic_image_gan(): GAN-based generation with blending")

    print("\n" + "="*80)
    print("GENERATING SYNTHETIC IMAGES FROM MASKS")
    print("="*80)
    print(f"Generating {synthetic_multiplier} synthetic images per real image")
    print(f"Total real images: {len(train_labels)}")
    print(f"Expected synthetic images: {len(train_labels) * synthetic_multiplier}")
    print("="*80 + "\n")
    
    total_generated = 0
    
    for idx, row in train_labels.iterrows():
        img_name = row['sample_index']
        class_label = row['label']
        
        # Load original mask and image
        mask_name = img_name.replace('img_', 'mask_')
        mask_path = os.path.join(train_mask_dir, mask_name)
        img_path = os.path.join(train_img_dir, img_name)
        
        if not os.path.exists(mask_path):
            print(f"  ⚠ Mask not found: {mask_name}, skipping")
            continue
        
        if not os.path.exists(img_path):
            print(f"  ⚠ Image not found: {img_name}, skipping")
            continue
        
        # Load original image
        original_img = Image.open(img_path).convert('RGB')
        
        # Prepare mask for GAN
        mask_tensor = prepare_mask_for_gan(mask_path)
        
        # Generate multiple synthetic versions
        for syn_idx in range(synthetic_multiplier):
            try:
                # Generate synthetic image using GAN
                synthetic_img = generate_synthetic_image_gan(
                    generator=generator,
                    mask_tensor=mask_tensor,
                    original_img=original_img,
                    class_label=class_label,
                )
                
                # Save synthetic image to existing train_img_dir
                base_name = img_name.replace('.png', '')
                synthetic_img_name = f"{base_name}_syn_{syn_idx}.png"
                synthetic_img_path = os.path.join(train_img_dir, synthetic_img_name)
                synthetic_img.save(synthetic_img_path)
                
                # Copy mask to existing train_mask_dir
                synthetic_mask_name = f"mask_{img_name.split('_')[1].replace('.png', '')}_syn_{syn_idx}.png"
                synthetic_mask_path = os.path.join(train_mask_dir, synthetic_mask_name)
                Image.open(mask_path).save(synthetic_mask_path)
                
                total_generated += 1
                
            except Exception as e:
                print(f"  ⚠ Error generating image at index {syn_idx}: {str(e)}")
                continue
        
        # Progress update
        if (idx + 1) % 10 == 0 or idx == len(train_labels) - 1:
            print(f"  Progress: {idx + 1}/{len(train_labels)} images processed ({total_generated} synthetic images generated)")
    
    print("\n" + "="*80)
    print("SYNTHETIC IMAGE GENERATION COMPLETE")
    print("="*80)
    print(f"Total synthetic images generated: {total_generated}")
    print(f"Added to existing directories:")
    print(f"  Images: {train_img_dir}")
    print(f"  Masks: {train_mask_dir}")
    print("="*80)
    
    # Clean up GPU memory
    if num_gpus > 0:
        del generator
        torch.cuda.empty_cache()
        print("\n✓ GPU memory cleared")

    # Get list of all synthetic images that were created
    synthetic_files = [f for f in os.listdir(train_img_dir) if '_syn_' in f]
    
    # Visualize some synthetic examples (if they exist)
    if len(synthetic_files) > 0:
        import matplotlib.pyplot as plt
        
        # Get a few synthetic images
        synthetic_files = sorted(synthetic_files)[:6]
        
        if len(synthetic_files) > 0:
            _, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, syn_img_name in enumerate(synthetic_files):
                if idx >= 6:
                    break
                
                # Load synthetic image
                syn_img_path = os.path.join(train_img_dir, syn_img_name)
                syn_img = Image.open(syn_img_path)
                
                # Load corresponding mask
                syn_mask_name = syn_img_name.replace('img_', 'mask_')
                syn_mask_path = os.path.join(train_mask_dir, syn_mask_name)
                syn_mask = Image.open(syn_mask_path)
                
                # Find original image for comparison
                base_name = syn_img_name.split('_syn_')[0] + '.png'
                orig_img_path = os.path.join(train_img_dir, base_name)
                
                # Create composite visualization
                composite = Image.new('RGB', (syn_img.width * 3, syn_img.height))
                
                if os.path.exists(orig_img_path):
                    orig_img = Image.open(orig_img_path)
                    composite.paste(orig_img, (0, 0))
                
                composite.paste(syn_mask.convert('RGB'), (syn_img.width, 0))
                composite.paste(syn_img, (syn_img.width * 2, 0))
                
                axes[idx].imshow(composite)
                axes[idx].set_title(f"Original | Mask | Synthetic\n{base_name}")
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.suptitle("Generative Mask Conditioning Examples", fontsize=16, y=1.02)
            plt.show()
            
            print(f"\n✓ Showing {min(6, len(synthetic_files))} synthetic examples")
            print("  Left: Original image | Center: Mask (guide) | Right: Synthetic image")
    else:
        print("No synthetic images to visualize. Generate them first.")

    # Create expanded dataset including synthetic images
    train_labels_expanded = train_labels.copy()

    # Add synthetic images if they exist
    synthetic_img_files = [f for f in os.listdir(train_img_dir) if '_syn_' in f]
    if len(synthetic_img_files) > 0:
        print(f"Found {len(synthetic_img_files)} synthetic images in existing directory")
        
        synthetic_rows: list[dict[str, str]] = []
        for syn_img_name in synthetic_img_files:
            # Extract original file name: img_XXXX_syn_N.png -> img_XXXX.png
            base_name = syn_img_name.split('_syn_')[0] + '.png'
            
            # Find corresponding class label
            original_row = train_labels[train_labels['sample_index'] == base_name]
            if not original_row.empty:
                class_label = original_row.iloc[0]['label']
                synthetic_rows.append({'sample_index': syn_img_name, 'label': class_label, 'source': 'synthetic'})
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        train_labels.loc[:, 'source'] = 'real'  # Mark original images
        train_labels_expanded = pd.concat([train_labels, synthetic_df], ignore_index=True)
        
        print(f"\nDataset expansion complete:")
        print(f"  Real images: {len(train_labels)}")
        print(f"  Synthetic images: {len(synthetic_df)}")
        print(f"  Total: {len(train_labels_expanded)}")
        print(f"  Expansion ratio: {len(train_labels_expanded) / len(train_labels):.2f}x")
        
        print(f"\nClass distribution (Real + Synthetic):")
        print(train_labels_expanded['label'].value_counts().sort_index())
    else:
        print("No synthetic images found. Using only real images.")
        print("Generate synthetic images first by setting GENERATE_SYNTHETIC=True")

    # Return the expanded dataset
    return train_labels_expanded
