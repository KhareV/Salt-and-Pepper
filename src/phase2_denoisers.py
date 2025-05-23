"""
QuadNet Phase 2: Ensemble of Classical Denoisers
Implementation of multiple denoising algorithms for salt-and-pepper noise.
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
from config import CONFIG

class DenoisingEnsemble:
    """Ensemble of classical denoising methods for salt-and-pepper noise."""
    
    def __init__(self):
        self.config = CONFIG
    
    def apply_all_denoisers(self, noisy_image: np.ndarray, 
                           noise_info: Dict) -> Dict[str, np.ndarray]:
        """
        Apply all denoising methods to the input image.
        
        Args:
            noisy_image: Noisy input image
            noise_info: Noise analysis from Phase 1
            
        Returns:
            Dictionary of denoised images from each method
        """
        results = {}
        
        # Get adaptive parameters
        adaptive_params = noise_info['adaptive_params']
        
        # 1. Median Filter (multiple kernel sizes)
        results['median_3x3'] = self.median_filter(noisy_image, 3)
        results['median_5x5'] = self.median_filter(noisy_image, 5)
        results['median_adaptive'] = self.median_filter(
            noisy_image, adaptive_params['median_kernel_size']
        )
        
        # 2. NAMF (Non-local Adaptive Mean Filter)
        results['namf'] = self.namf_filter(noisy_image, noise_info)
        
        # 3. Morphological Operations
        results['morphological'] = self.morphological_filter(noisy_image, noise_info)
        
        # 4. Bilateral Filter
        results['bilateral'] = self.bilateral_filter(
            noisy_image, adaptive_params['bilateral_d']
        )
        
        # 5. Adaptive Hybrid Filter
        results['adaptive_hybrid'] = self.adaptive_hybrid_filter(noisy_image, noise_info)
        
        # 6. Iterative Median Filter
        results['iterative_median'] = self.iterative_median_filter(noisy_image, noise_info)
        
        return results
    
    def median_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply median filter with specified kernel size."""
        return cv2.medianBlur(
            (image * 255).astype(np.uint8), kernel_size
        ).astype(np.float64) / 255.0
    
    def namf_filter(self, image: np.ndarray, noise_info: Dict) -> np.ndarray:
        """
        Non-local Adaptive Mean Filter implementation.
        Adapted for salt-and-pepper noise removal.
        """
        h, w = image.shape
        result = image.copy()
        window_size = CONFIG.NAMF_WINDOW_SIZE
        search_window = CONFIG.NAMF_SEARCH_WINDOW
        noise_mask = noise_info['noise_mask']
        
        # Pad image for border handling
        pad_size = search_window // 2
        padded_image = np.pad(image, pad_size, mode='reflect')
        padded_result = np.pad(result, pad_size, mode='reflect')
        
        # Process only noisy pixels for efficiency
        noisy_pixels = np.where(noise_mask)
        
        for idx in range(len(noisy_pixels[0])):
            i, j = noisy_pixels[0][idx], noisy_pixels[1][idx]
            
            # Adjust coordinates for padded image
            pi, pj = i + pad_size, j + pad_size
            
            # Extract reference patch
            ref_patch = padded_image[
                pi - window_size//2 : pi + window_size//2 + 1,
                pj - window_size//2 : pj + window_size//2 + 1
            ]
            
            weights = []
            values = []
            
            # Search in neighborhood
            for di in range(-search_window//2, search_window//2 + 1):
                for dj in range(-search_window//2, search_window//2 + 1):
                    if di == 0 and dj == 0:
                        continue
                    
                    # Check bounds
                    ni, nj = pi + di, pj + dj
                    if (ni - window_size//2 < 0 or 
                        ni + window_size//2 + 1 >= padded_image.shape[0] or
                        nj - window_size//2 < 0 or 
                        nj + window_size//2 + 1 >= padded_image.shape[1]):
                        continue
                    
                    # Extract comparison patch
                    comp_patch = padded_image[
                        ni - window_size//2 : ni + window_size//2 + 1,
                        nj - window_size//2 : nj + window_size//2 + 1
                    ]
                    
                    # Calculate similarity (using robust distance)
                    if ref_patch.shape == comp_patch.shape:
                        # Use median-based distance for robustness
                        diff = np.abs(ref_patch - comp_patch)
                        distance = np.median(diff)
                        weight = np.exp(-distance / 0.1)  # Bandwidth parameter
                        
                        weights.append(weight)
                        values.append(padded_image[ni, nj])
            
            # Compute weighted average
            if weights:
                weights = np.array(weights)
                values = np.array(values)
                weights = weights / np.sum(weights)
                
                new_value = np.sum(weights * values)
                padded_result[pi, pj] = new_value
        
        # Remove padding
        result = padded_result[pad_size:-pad_size, pad_size:-pad_size]
        return result
    
    def morphological_filter(self, image: np.ndarray, noise_info: Dict) -> np.ndarray:
        """Apply morphological operations for impulse noise removal."""
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create morphological kernel
        kernel_size = CONFIG.MORPH_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply opening (erosion followed by dilation)
        opened = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
        
        # Apply closing (dilation followed by erosion)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return result.astype(np.float64) / 255.0
    
    def bilateral_filter(self, image: np.ndarray, d: int) -> np.ndarray:
        """Apply bilateral filter for edge-preserving smoothing."""
        img_uint8 = (image * 255).astype(np.uint8)
        
        result = cv2.bilateralFilter(
            img_uint8, d, 
            CONFIG.BILATERAL_SIGMA_COLOR,
            CONFIG.BILATERAL_SIGMA_SPACE
        )
        
        return result.astype(np.float64) / 255.0
    
    def adaptive_hybrid_filter(self, image: np.ndarray, noise_info: Dict) -> np.ndarray:
        """
        Adaptive hybrid filter that combines multiple methods based on local characteristics.
        """
        h, w = image.shape
        result = image.copy()
        noise_mask = noise_info['noise_mask']
        confidence_map = noise_info['confidence_map']
        
        # Apply different filters based on confidence levels
        high_confidence = confidence_map > 0.8
        medium_confidence = (confidence_map > 0.4) & (confidence_map <= 0.8)
        low_confidence = confidence_map <= 0.4
        
        # High confidence: strong median filter
        if np.any(high_confidence):
            median_strong = self.median_filter(image, 5)
            result[high_confidence] = median_strong[high_confidence]
        
        # Medium confidence: bilateral filter
        if np.any(medium_confidence):
            bilateral = self.bilateral_filter(image, 5)
            result[medium_confidence] = bilateral[medium_confidence]
        
        # Low confidence: mild smoothing
        if np.any(low_confidence):
            gaussian = cv2.GaussianBlur(image, (3, 3), 0.5)
            result[low_confidence] = gaussian[low_confidence]
        
        return result
    
    def iterative_median_filter(self, image: np.ndarray, noise_info: Dict, 
                              max_iterations: int = 3) -> np.ndarray:
        """
        Iterative median filter that applies median filtering multiple times
        with decreasing kernel sizes.
        """
        result = image.copy()
        noise_density = noise_info['noise_density']
        
        # Determine number of iterations based on noise density
        num_iterations = min(max_iterations, int(noise_density * 10) + 1)
        
        # Start with larger kernel and reduce size
        kernel_sizes = [7, 5, 3][:num_iterations]
        
        for kernel_size in kernel_sizes:
            result = self.median_filter(result, kernel_size)
            
            # Early stopping if improvement is minimal
            if num_iterations > 1:
                improvement = np.mean(np.abs(result - image))
                if improvement < 0.01:  # Threshold for improvement
                    break
        
        return result