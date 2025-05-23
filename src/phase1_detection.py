"""
QuadNet Phase 1: Noise Detection and Characterization
Identifies salt-and-pepper noise characteristics in the input image.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Dict
from config import CONFIG

class NoiseDetector:
    """Salt-and-pepper noise detection and characterization."""
    
    def __init__(self):
        self.salt_threshold = CONFIG.SALT_THRESHOLD / 255.0
        self.pepper_threshold = CONFIG.PEPPER_THRESHOLD / 255.0
        self.window_size = CONFIG.NOISE_WINDOW_SIZE
    
    def detect_salt_pepper_noise(self, image: np.ndarray) -> Dict:
        """
        Detect salt-and-pepper noise in the input image.
        
        Args:
            image: Input grayscale image (0-1 range)
            
        Returns:
            Dictionary containing noise analysis results
        """
        # Convert to uint8 for processing
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Detect salt and pepper pixels
        salt_mask = image >= self.salt_threshold
        pepper_mask = image <= self.pepper_threshold
        noise_mask = salt_mask | pepper_mask
        
        # Calculate noise density
        total_pixels = image.size
        salt_pixels = np.sum(salt_mask)
        pepper_pixels = np.sum(pepper_mask)
        total_noise_pixels = np.sum(noise_mask)
        
        noise_density = total_noise_pixels / total_pixels
        salt_ratio = salt_pixels / max(total_noise_pixels, 1)
        pepper_ratio = pepper_pixels / max(total_noise_pixels, 1)
        
        # Analyze spatial distribution
        spatial_distribution = self._analyze_spatial_distribution(noise_mask)
        
        # Detect isolated vs clustered noise
        clustering_info = self._analyze_noise_clustering(noise_mask)
        
        # Create noise map with confidence levels
        confidence_map = self._create_confidence_map(image, noise_mask)
        
        return {
            'noise_mask': noise_mask,
            'salt_mask': salt_mask,
            'pepper_mask': pepper_mask,
            'noise_density': noise_density,
            'salt_ratio': salt_ratio,
            'pepper_ratio': pepper_ratio,
            'spatial_distribution': spatial_distribution,
            'clustering_info': clustering_info,
            'confidence_map': confidence_map,
            'adaptive_params': self._suggest_adaptive_parameters(noise_density)
        }
    
    def _analyze_spatial_distribution(self, noise_mask: np.ndarray) -> Dict:
        """Analyze spatial distribution of noise."""
        h, w = noise_mask.shape
        
        # Divide image into quadrants and analyze noise distribution
        quad_h, quad_w = h // 2, w // 2
        quadrants = [
            noise_mask[:quad_h, :quad_w],           # Top-left
            noise_mask[:quad_h, quad_w:],           # Top-right
            noise_mask[quad_h:, :quad_w],           # Bottom-left
            noise_mask[quad_h:, quad_w:]            # Bottom-right
        ]
        
        quad_densities = [np.mean(quad) for quad in quadrants]
        uniformity = 1.0 - np.std(quad_densities) / (np.mean(quad_densities) + 1e-8)
        
        return {
            'quadrant_densities': quad_densities,
            'uniformity_score': uniformity,
            'max_density_quadrant': np.argmax(quad_densities)
        }
    
    def _analyze_noise_clustering(self, noise_mask: np.ndarray) -> Dict:
        """Analyze clustering patterns in noise distribution."""
        # Use morphological operations to detect clusters
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilate to connect nearby noise pixels
        dilated = cv2.dilate(noise_mask.astype(np.uint8), kernel, iterations=1)
        
        # Find connected components
        num_components, labels = cv2.connectedComponents(dilated)
        
        # Analyze component sizes
        component_sizes = []
        for i in range(1, num_components):  # Skip background (0)
            size = np.sum(labels == i)
            component_sizes.append(size)
        
        if component_sizes:
            avg_cluster_size = np.mean(component_sizes)
            max_cluster_size = np.max(component_sizes)
            clustering_ratio = len(component_sizes) / max(np.sum(noise_mask), 1)
        else:
            avg_cluster_size = 0
            max_cluster_size = 0
            clustering_ratio = 0
        
        return {
            'num_clusters': len(component_sizes),
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size,
            'clustering_ratio': clustering_ratio
        }
    
    def _create_confidence_map(self, image: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        """Create confidence map for noise detection."""
        confidence = np.zeros_like(image)
        
        # High confidence for extreme values
        confidence[image >= self.salt_threshold] = 1.0
        confidence[image <= self.pepper_threshold] = 1.0
        
        # Medium confidence for values close to extremes
        salt_buffer = (self.salt_threshold + 1.0) / 2
        pepper_buffer = self.pepper_threshold / 2
        
        medium_salt = (image >= salt_buffer) & (image < self.salt_threshold)
        medium_pepper = (image <= pepper_buffer) & (image > self.pepper_threshold)
        
        confidence[medium_salt] = 0.6
        confidence[medium_pepper] = 0.6
        
        # Apply Gaussian smoothing for spatial consistency
        confidence = cv2.GaussianBlur(confidence, (5, 5), 1.0)
        
        return confidence
    
    def _suggest_adaptive_parameters(self, noise_density: float) -> Dict:
        """Suggest adaptive parameters based on noise characteristics."""
        if noise_density < 0.05:
            # Low noise: use smaller kernels, less aggressive filtering
            return {
                'median_kernel_size': 3,
                'bilateral_d': 5,
                'morphology_iterations': 1,
                'namf_window_size': 5
            }
        elif noise_density < 0.15:
            # Medium noise: standard parameters
            return {
                'median_kernel_size': 5,
                'bilateral_d': 9,
                'morphology_iterations': 2,
                'namf_window_size': 7
            }
        else:
            # High noise: more aggressive filtering
            return {
                'median_kernel_size': 7,
                'bilateral_d': 13,
                'morphology_iterations': 3,
                'namf_window_size': 9
            }