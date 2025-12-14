import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

def focus_filter(train_img_dir: str, train_mask_dir: str, test_img_dir: str, test_mask_dir: str, train_labels: pd.DataFrame, min_size: tuple[int, int]=(128, 128)):
    """
    Applica una mask di focus alle immagini di training e test.
    Le immagini mascherate vengono salvate in nuove cartelle.
    """

    # Remove any previously masked&cropped images from the training set
    train_labels = train_labels[~train_labels['sample_index'].str.contains('_crop_')].reset_index(drop=True)
    for img_file in os.listdir(train_img_dir):
        if '_crop_' in img_file:
            os.remove(os.path.join(train_img_dir, img_file))
    for mask_file in os.listdir(train_mask_dir):
        if '_crop_' in mask_file:
            os.remove(os.path.join(train_mask_dir, mask_file))
    
    def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        image: PIL RGB image
        mask: PIL grayscale mask (0 = nero, 255 = bianco)
        return: masked image (PIL)
        """
        
        # Resize mask to match image size if they differ
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.BILINEAR)

        # Converti in numpy
        img_np = np.array(image).astype(np.uint8)
        mask_np = np.array(mask).astype(np.uint8)

        # Normalizza la mask a 0–1
        mask_np = mask_np / 255.0

        # Se l’immagine ha 3 canali, estendi la mask
        if img_np.ndim == 3:
            mask_np = np.expand_dims(mask_np, axis=-1)

        # Moltiplica → le zone nere diventano 0 (nero)
        masked_img_np = (img_np * mask_np).astype(np.uint8)

        return Image.fromarray(masked_img_np)

    # === VISUALIZZA A VIDEO ALCUNI ESEMPI ===
    samples = sorted(os.listdir(train_img_dir))[:4]  # primi 4 esempi

    _, axes = plt.subplots(len(samples), 2, figsize=(12, 2 * len(samples)))

    for i, img_name in enumerate(samples):

        # Carica immagine e mask corrispondente
        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, img_name.replace("img_", "mask_"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Applica la mask
        masked_image = apply_mask(image, mask)

        # --- Plot ---
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masked_image)
        axes[i, 1].set_title("Masked Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Apply masking to all training images and replace original files
    train_img_files = sorted(os.listdir(train_img_dir))

    for img_name in tqdm(train_img_files, desc="Processing training images"):
        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, img_name.replace("img_", "mask_"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        masked_image = apply_mask(image, mask)
        masked_image.save(img_path)  # Overwrite original file
    print("Training images masked and replaced.")

    # Apply masking to all test images and replace original files
    test_img_files = sorted(os.listdir(test_img_dir))
    for img_name in tqdm(test_img_files, desc="Processing test images"):
        img_path = os.path.join(test_img_dir, img_name)
        mask_path = os.path.join(test_mask_dir, img_name.replace("img_", "mask_"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        masked_image = apply_mask(image, mask)
        masked_image.save(img_path)  # Overwrite original file

    print("Test images masked and replaced.")

    # Crop masked images to the extent of information
    # Now that we have applied the masks, we can crop the images to the bounding box of the non-zero regions in the masks.
    def crop_to_mask(image: Image.Image) -> Image.Image:
        """
        Crop the image to the bounding box of the non-zero regions.
        Since the mask has already been applied to the image, we detect non-zero pixels directly from the image.
        If the cropped image is smaller than min_size, add zero-padding to reach min_size.
        
        image: PIL RGB image (already masked, with black background)
        mask: PIL grayscale mask (not used, kept for compatibility)
        min_size: tuple (width, height) - minimum size for the output image
        return: cropped and padded image (PIL)
        """
        # Convert image to numpy array
        img_np = np.array(image).astype(np.uint8)
        
        # Find non-zero pixels in any channel (R, G, or B)
        # Sum across color channels and check where sum > 0
        non_zero_mask = np.sum(img_np, axis=2) > 0
        coords = np.column_stack(np.where(non_zero_mask))
        
        if coords.size == 0:
            return image  # No cropping if image is completely black
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1  # add 1 to include the max pixel
        
        cropped = image.crop((x_min, y_min, x_max, y_max))
        
        # Check if padding is needed
        width, height = cropped.size
        min_width, min_height = min_size
        
        if width < min_width or height < min_height:
            # Calculate padding needed
            pad_width = max(0, min_width - width)
            pad_height = max(0, min_height - height)
            
            # Create new image with black padding
            padded = Image.new('RGB', (max(width, min_width), max(height, min_height)), (0, 0, 0))
            
            # Paste cropped image in the center
            paste_x = pad_width // 2
            paste_y = pad_height // 2
            padded.paste(cropped, (paste_x, paste_y))
            
            return padded
        
        return cropped

    # === VISUALIZZA A VIDEO ALCUNI ESEMPI DI CROP ===
    samples = sorted(os.listdir(train_img_dir))[:4]  # primi 4 esempi
    _, axes = plt.subplots(len(samples), 3, figsize=(12, 3 * len(samples)))

    for i, img_name in enumerate(samples):
        # Carica immagine e mask corrispondente
        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, img_name.replace("img_", "mask_"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Applica la mask
        masked_image = apply_mask(image, mask)

        # Esegui il crop con padding se necessario
        cropped_image = crop_to_mask(masked_image)

        # --- Plot ---
        axes[i, 0].imshow(masked_image)
        axes[i, 0].set_title("Masked Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cropped_image)
        axes[i, 2].set_title(f"Cropped Image\n{cropped_image.size[0]}x{cropped_image.size[1]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # Crop all images and save to same directories

    # Process training images
    for img_name in tqdm(os.listdir(train_img_dir), desc="Processing training images"):
        img_path = os.path.join(train_img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        cropped_image = crop_to_mask(image)

        # Create new filename with '_crop_' before extension
        name_parts = os.path.splitext(img_name)
        new_img_name = name_parts[0] + '_crop_' + name_parts[1]
        new_img_path = os.path.join(train_img_dir, new_img_name)
        
        cropped_image.save(new_img_path)
        os.remove(img_path)  # Remove old file
    print("\nCropped training images saved.")
    # Process test images
    for img_name in tqdm(os.listdir(test_img_dir), desc="Processing test images"):
        img_path = os.path.join(test_img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        cropped_image = crop_to_mask(image)

        # Create new filename with '_crop_' before extension
        name_parts = os.path.splitext(img_name)
        new_img_name = name_parts[0] + '_crop_' + name_parts[1]
        new_img_path = os.path.join(test_img_dir, new_img_name)
        
        cropped_image.save(new_img_path)
        os.remove(img_path)  # Remove old file
    print("\nCropped test images saved.")