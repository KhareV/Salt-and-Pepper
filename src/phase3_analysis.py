"""
QuadNet Phase 3: Statistical Signature Extraction
Extract statistical signatures from denoised images for quality assessment.
"""

import numpy as np
import cv2
import pywt
from scipy import ndimage
from scipy.fft import fft2, fftshift
from typing import Dict, Tuple
from config import CONFIG

class StatisticalAnalyzer:
    """Extract statistical signatures for denoising quality assessment."""
    
    def __init__(self):
        self.config = CONFIG
    
    def extract_signatures(self, original: np.ndarray, denoised: np.ndarray) -> Dict:
        """
        Extract comprehensive statistical signatures from denoised image.
        
        Args:
            original: Original clean image (if available)
            denoised: Denoised image
            
        Returns:
            Dictionary containing various statistical measures
        """
        signatures = {}
        
        # 1. Power Spectral Density Analysis
        signatures['psd'] = self._analyze_power_spectrum(denoised)
        
        # 2. Wavelet Energy Distribution
        signatures['wavelet'] = self._analyze_wavelet_energy(denoised)
        
        # 3. Local Variance Analysis
        signatures['local_variance'] = self._analyze_local_variance(denoised)
        
        # 4. Edge Preservation Metrics
        if original is not None:
            signatures['edge_preservation'] = self._analyze_edge_preservation(
                original, denoised
            )
        
        # 5. Texture Analysis
        signatures['texture'] = self._analyze_texture_features(denoised)
        
        # 6. Noise Residual Analysis
        if original is not None:
            signatures['noise_residual'] = self._analyze_noise_residual(
                original, denoised
            )
        
        # 7. Gradient Analysis
        signatures['gradient'] = self._analyze_gradient_statistics(denoised)
        
        return signatures
    
    def _analyze_power_spectrum(self, image: np.ndarray) -> Dict:
        """Analyze power spectral density of the image."""
        # Compute 2D FFT
        fft = fft2(image)
        psd = np.abs(fftshift(fft)) ** 2
        
        # Normalize
        psd = psd / np.sum(psd)
        
        # Extract radial profile
        center = np.array(psd.shape) // 2
        y, x = np.ogrid[:psd.shape[0], :psd.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Compute radial average
        r_int = r.astype(int)
        tbin = np.bincount(r_int.ravel(), psd.ravel())
        nr = np.bincount(r_int.ravel())
        radial_profile = tbin / (nr + 1e-8)
        
        # Extract frequency characteristics
        high_freq_energy = np.sum(radial_profile[len(radial_profile)//2:])
        low_freq_energy = np.sum(radial_profile[:len(radial_profile)//2])
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        # Spectral centroid
        frequencies = np.arange(len(radial_profile))
        spectral_centroid = np.sum(frequencies * radial_profile) / np.sum(radial_profile)
        
        return {
            'radial_profile': radial_profile,
            'high_freq_energy': high_freq_energy,
            'low_freq_energy': low_freq_energy,
            'freq_ratio': freq_ratio,
            'spectral_centroid': spectral_centroid,
            'total_energy': np.sum(psd)
        }
    
    def _analyze_wavelet_energy(self, image: np.ndarray) -> Dict:
        """Analyze energy distribution in wavelet domain."""
        wavelet_type = CONFIG.WAVELET_TYPE
        levels = CONFIG.WAVELET_LEVELS
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet_type, level=levels)
        
        # Extract energy from each subband
        energies = {}
        
        # Approximation coefficients (low-low)
        energies['LL'] = np.sum(coeffs[0] ** 2)
        
        # Detail coefficients for each level
        for i, (LH, HL, HH) in enumerate(coeffs[1:]):
            level = i + 1
            energies[f'LH_{level}'] = np.sum(LH ** 2)
            energies[f'HL_{level}'] = np.sum(HL ** 2)
            energies[f'HH_{level}'] = np.sum(HH ** 2)
        
        # Normalize energies
        total_energy = sum(energies.values())
        normalized_energies = {k: v / total_energy for k, v in energies.items()}
        
        # Compute energy ratios
        high_freq_energy = sum(v for k, v in normalized_energies.items() if 'HH' in k)
        low_freq_energy = normalized_energies['LL']
        energy_concentration = max(normalized_energies.values())
        
        return {
            'energies': energies,
            'normalized_energies': normalized_energies,
            'high_freq_energy': high_freq_energy,
            'low_freq_energy': low_freq_energy,
            'energy_concentration': energy_concentration,
            'total_energy': total_energy
        }
    
    def _analyze_local_variance(self, image: np.ndarray) -> Dict:
        """Analyze local variance characteristics."""
        # Compute local variance using different window sizes
        variances = {}
        window_sizes = [3, 5, 7, 9]
        
        for ws in window_sizes:
            # Local mean
            kernel = np.ones((ws, ws)) / (ws * ws)
            local_mean = cv2.filter2D(image, -1, kernel)
            
            # Local variance
            local_variance = cv2.filter2D(image**2, -1, kernel) - local_mean**2
            
            variances[f'var_{ws}x{ws}'] = {
                'mean': np.mean(local_variance),
                'std': np.std(local_variance),
                'max': np.max(local_variance),
                'percentile_95': np.percentile(local_variance, 95)
            }
        
        # Global variance
        global_variance = np.var(image)
        
        return {
            'local_variances': variances,
            'global_variance': global_variance
        }
    
    def _analyze_edge_preservation(self, original: np.ndarray, 
                                 denoised: np.ndarray) -> Dict:
        """Analyze how well edges are preserved during denoising."""
        # Compute edges using Canny edge detector
        original_uint8 = (original * 255).astype(np.uint8)
        denoised_uint8 = (denoised * 255).astype(np.uint8)
        
        # Edge detection
        edges_original = cv2.Canny(original_uint8, 50, 150)
        edges_denoised = cv2.Canny(denoised_uint8, 50, 150)
        
        # Edge preservation metrics
        edges_original_norm = edges_original.astype(np.float64) / 255.0
        edges_denoised_norm = edges_denoised.astype(np.float64) / 255.0
        
        # True positive rate (preserved edges)
        true_positive = np.sum(edges_original_norm * edges_denoised_norm)
        total_original_edges = np.sum(edges_original_norm)
        edge_preservation_rate = true_positive / (total_original_edges + 1e-8)
        
        # False positive rate (spurious edges)
        false_positive = np.sum(edges_denoised_norm * (1 - edges_original_norm))
        total_denoised_edges = np.sum(edges_denoised_norm)
        spurious_edge_rate = false_positive / (total_denoised_edges + 1e-8)
        
        # Edge strength correlation
        # Compute gradient magnitudes
        grad_x_orig = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_orig = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
        
        grad_x_den = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_den = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_den = np.sqrt(grad_x_den**2 + grad_y_den**2)
        
        # Correlation coefficient
        correlation = np.corrcoef(grad_mag_orig.flatten(), grad_mag_den.flatten())[0, 1]
        
        return {
            'edge_preservation_rate': edge_preservation_rate,
            'spurious_edge_rate': spurious_edge_rate,
            'gradient_correlation': correlation,
            'total_original_edges': total_original_edges,
            'total_denoised_edges': total_denoised_edges
        }
    
    def _analyze_texture_features(self, image: np.ndarray) -> Dict:
        """Analyze texture characteristics using various descriptors."""
        # Local Binary Pattern (simplified version)
        def local_binary_pattern(img, radius=1):
            h, w = img.shape
            lbp = np.zeros_like(img)
            
            # Define neighbor coordinates
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                        (1, 1), (1, 0), (1, -1), (0, -1)]
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = img[i, j]
                    pattern = 0
                    
                    for k, (di, dj) in enumerate(neighbors):
                        if img[i + di, j + dj] >= center:
                            pattern |= (1 << k)
                    
                    lbp[i, j] = pattern
            
            return lbp
        
        # Compute LBP
        img_uint8 = (image * 255).astype(np.uint8)
        lbp = local_binary_pattern(img_uint8)
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        # Texture energy and entropy
        energy = np.sum(lbp_hist**2)
        entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
        
        # Co-occurrence matrix features (simplified)
        # Horizontal co-occurrence
        glcm = np.zeros((256, 256))
        img_quantized = (image * 255).astype(int)
        
        for i in range(img_quantized.shape[0]):
            for j in range(img_quantized.shape[1] - 1):
                glcm[img_quantized[i, j], img_quantized[i, j + 1]] += 1
        
        # Normalize GLCM
        glcm = glcm / np.sum(glcm)
        
        # GLCM features
        contrast = 0
        homogeneity = 0
        for i in range(256):
            for j in range(256):
                contrast += glcm[i, j] * (i - j)**2
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        
        return {
            'lbp_energy': energy,
            'lbp_entropy': entropy,
            'glcm_contrast': contrast,
            'glcm_homogeneity': homogeneity,
            'lbp_histogram': lbp_hist
        }
    
    def _analyze_noise_residual(self, original: np.ndarray, 
                               denoised: np.ndarray) -> Dict:
        """Analyze the noise residual (difference between original and denoised)."""
        residual = original - denoised
        
        # Statistical properties of residual
        residual_stats = {
            'mean': np.mean(residual),
            'std': np.std(residual),
            'skewness': self._calculate_skewness(residual),
            'kurtosis': self._calculate_kurtosis(residual),
            'max_abs': np.max(np.abs(residual))
        }
        
        # Residual autocorrelation (measure of spatial correlation)
        autocorr = self._calculate_autocorrelation(residual)
        
        return {
            'statistics': residual_stats,
            'autocorrelation': autocorr,
            'residual_image': residual
        }
    
    def _analyze_gradient_statistics(self, image: np.ndarray) -> Dict:
        """Analyze gradient statistics of the image."""
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Statistics
        grad_stats = {
            'mean_magnitude': np.mean(grad_magnitude),
            'std_magnitude': np.std(grad_magnitude),
            'max_magnitude': np.max(grad_magnitude),
            'directional_variance': np.var(grad_direction)
        }
        
        # Gradient histogram
        mag_hist, _ = np.histogram(grad_magnitude, bins=50, density=True)
        
        return {
            'statistics': grad_stats,
            'magnitude_histogram': mag_hist,
            'mean_direction': np.mean(grad_direction)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std)**3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std)**4) - 3
    
    def _calculate_autocorrelation(self, image: np.ndarray, max_lag: int = 5) -> Dict:
        """Calculate spatial autocorrelation of image."""
        correlations = {}
        
        for lag in range(1, max_lag + 1):
            # Horizontal correlation
            if lag < image.shape[1]:
                h_corr = np.corrcoef(
                    image[:, :-lag].flatten(),
                    image[:, lag:].flatten()
                )[0, 1]
                correlations[f'horizontal_lag_{lag}'] = h_corr
            
            # Vertical correlation
            if lag < image.shape[0]:
                v_corr = np.corrcoef(
                    image[:-lag, :].flatten(),
                    image[lag:, :].flatten()
                )[0, 1]
                correlations[f'vertical_lag_{lag}'] = v_corr
        
        return correlations