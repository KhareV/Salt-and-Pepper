"""
QuadNet Configuration Module
Contains all hyperparameters and settings for the QuadNet framework.
"""

import numpy as np

class QuadNetConfig:
    """Configuration class for QuadNet parameters."""
    
    # Phase 1: Noise Detection Parameters
    SALT_THRESHOLD = 245  # Threshold for salt noise detection
    PEPPER_THRESHOLD = 10  # Threshold for pepper noise detection
    NOISE_WINDOW_SIZE = 3  # Window size for local noise analysis
    
    # Phase 2: Denoiser Parameters
    MEDIAN_KERNEL_SIZES = [3, 5, 7]  # Multiple kernel sizes for median filter
    BILATERAL_D = 9  # Diameter for bilateral filter
    BILATERAL_SIGMA_COLOR = 75  # Sigma color for bilateral filter
    BILATERAL_SIGMA_SPACE = 75  # Sigma space for bilateral filter
    MORPH_KERNEL_SIZE = 3  # Kernel size for morphological operations
    
    # NAMF Parameters
    NAMF_WINDOW_SIZE = 7  # Window size for NAMF
    NAMF_SIMILARITY_THRESHOLD = 0.8  # Similarity threshold for NAMF
    NAMF_SEARCH_WINDOW = 21  # Search window size for NAMF
    
    # Phase 3: Statistical Analysis Parameters
    WAVELET_TYPE = 'db4'  # Wavelet type for decomposition
    WAVELET_LEVELS = 3  # Number of decomposition levels
    FFT_FREQ_BINS = 64  # Number of frequency bins for FFT analysis
    
    # Phase 4: Scoring Weights
    SCORING_WEIGHTS = {
        'psnr': 0.3,
        'ssim': 0.25,
        'fft': 0.2,
        'edge': 0.15,
        'texture': 0.1
    }
    
    # Image Processing Parameters
    MAX_IMAGE_SIZE = 1024  # Maximum image dimension for processing
    EDGE_THRESHOLD = 0.1  # Threshold for edge detection
    
    # Output Parameters
    SAVE_INTERMEDIATE = True  # Save intermediate results
    VERBOSE = True  # Enable verbose output

# Global configuration instance
CONFIG = QuadNetConfig()