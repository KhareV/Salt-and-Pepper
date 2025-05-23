"""
QuadNet Utility Functions
Common utility functions used across the QuadNet framework.
"""

import os
import numpy as np
import cv2
from scipy import ndimage
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from typing import Tuple, Union, List
import os

def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image for QuadNet processing.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Normalized grayscale image as float array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to float and normalize
    image = img_as_float(image)
    return image

def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, 
                         salt_vs_pepper: float = 0.5) -> np.ndarray:
    """
    Add salt-and-pepper noise to an image.
    
    Args:
        image: Input image
        amount: Proportion of image pixels to replace with noise
        salt_vs_pepper: Proportion of salt vs pepper noise
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    
    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 1.0
    
    # Add pepper noise  
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0.0
    
    return noisy

def calculate_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    return peak_signal_noise_ratio(original, denoised, data_range=1.0)

def calculate_ssim(original: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    return structural_similarity(original, denoised, data_range=1.0)

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_as_ubyte(image))

def create_comparison_plot(original: np.ndarray, noisy: np.ndarray, 
                          results: dict, output_path: str) -> None:
    """
    Create comparison plot of denoising results.
    
    Args:
        original: Original clean image
        noisy: Noisy input image
        results: Dictionary containing denoised results and scores
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original and noisy images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    # Best result
    best_method = results['best_method']
    best_image = results['denoised_images'][best_method]
    axes[0, 2].imshow(best_image, cmap='gray')
    axes[0, 2].set_title(f'QuadNet Result ({best_method})')
    axes[0, 2].axis('off')
    
    # Individual method results
    methods = list(results['denoised_images'].keys())[:3]
    for i, method in enumerate(methods):
        axes[1, i].imshow(results['denoised_images'][method], cmap='gray')
        score = results['scores'][method]
        axes[1, i].set_title(f'{method.title()}\nScore: {score:.3f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def resize_image_if_needed(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image if it exceeds maximum size while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image