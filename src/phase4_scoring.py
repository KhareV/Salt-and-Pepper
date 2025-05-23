"""
QuadNet Phase 4: Performance Scoring and Selection
Evaluate and select the best denoising result based on multiple criteria.
"""
import cv2  # <- This import was missing!

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import CONFIG
from src.utils import calculate_psnr, calculate_ssim

class PerformanceScorer:
    """Performance scoring and selection for denoising results."""
    
    def __init__(self):
        self.weights = CONFIG.SCORING_WEIGHTS
    
    def evaluate_all_methods(self, original: Optional[np.ndarray],
                            noisy: np.ndarray,
                            denoised_images: Dict[str, np.ndarray],
                            statistical_signatures: Dict[str, Dict]) -> Dict:
        """
        Evaluate all denoising methods and select the best one.
        
        Args:
            original: Original clean image (may be None for real-world scenarios)
            noisy: Noisy input image
            denoised_images: Dictionary of denoised results
            statistical_signatures: Statistical analysis results
            
        Returns:
            Dictionary containing scores and best method selection
        """
        scores = {}
        detailed_metrics = {}
        
        for method_name, denoised_img in denoised_images.items():
            # Calculate individual metrics
            metrics = self._calculate_individual_metrics(
                original, noisy, denoised_img, 
                statistical_signatures.get(method_name, {})
            )
            
            # Compute weighted score
            total_score = self._compute_weighted_score(metrics)
            
            scores[method_name] = total_score
            detailed_metrics[method_name] = metrics
        
        # Select best method
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        # Generate ranking
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'scores': scores,
            'detailed_metrics': detailed_metrics,
            'best_method': best_method,
            'ranking': ranking,
            'score_analysis': self._analyze_score_distribution(scores)
        }
    
    def _calculate_individual_metrics(self, original: Optional[np.ndarray],
                                    noisy: np.ndarray,
                                    denoised: np.ndarray,
                                    signatures: Dict) -> Dict:
        """Calculate individual performance metrics."""
        metrics = {}
        
        # 1. Image Quality Metrics (if original is available)
        if original is not None:
            metrics['psnr'] = calculate_psnr(original, denoised)
            metrics['ssim'] = calculate_ssim(original, denoised)
            
            # Mean Squared Error
            metrics['mse'] = np.mean((original - denoised) ** 2)
            
            # Mean Absolute Error
            metrics['mae'] = np.mean(np.abs(original - denoised))
        else:
            # No-reference metrics based on statistical properties
            metrics['psnr'] = self._estimate_psnr_no_reference(noisy, denoised)
            metrics['ssim'] = self._estimate_ssim_no_reference(denoised)
            metrics['mse'] = 0.0  # Placeholder
            metrics['mae'] = 0.0  # Placeholder
        
        # 2. FFT-based Score
        metrics['fft'] = self._calculate_fft_score(denoised, signatures)
        
        # 3. Edge Preservation Score
        if original is not None:
            metrics['edge'] = self._calculate_edge_score(original, denoised)
        else:
            metrics['edge'] = self._estimate_edge_score_no_reference(denoised)
        
        # 4. Texture Preservation Score
        metrics['texture'] = self._calculate_texture_score(denoised, signatures)
        
        # 5. Noise Reduction Effectiveness
        metrics['noise_reduction'] = self._calculate_noise_reduction_score(
            noisy, denoised
        )
        
        # 6. Artifact Detection Score
        metrics['artifact'] = self._calculate_artifact_score(denoised)
        
        # 7. Smoothness Score
        metrics['smoothness'] = self._calculate_smoothness_score(denoised)
        
        return metrics
    
    def _compute_weighted_score(self, metrics: Dict) -> float:
        """Compute weighted total score from individual metrics."""
        # Normalize metrics to [0, 1] range
        normalized_metrics = self._normalize_metrics(metrics)
        
        # Apply weights
        weighted_score = (
            self.weights['psnr'] * normalized_metrics['psnr'] +
            self.weights['ssim'] * normalized_metrics['ssim'] +
            self.weights['fft'] * normalized_metrics['fft'] +
            self.weights['edge'] * normalized_metrics['edge'] +
            self.weights['texture'] * normalized_metrics['texture']
        )
        
        # Add bonus for low artifacts and good noise reduction
        weighted_score += 0.05 * normalized_metrics['noise_reduction']
        weighted_score -= 0.05 * normalized_metrics['artifact']
        
        return max(0.0, min(1.0, weighted_score))  # Clamp to [0, 1]
    
    def _normalize_metrics(self, metrics: Dict) -> Dict:
        """Normalize metrics to [0, 1] range."""
        normalized = {}
        
        # PSNR: typically 10-50 dB, normalize to [0, 1]
        normalized['psnr'] = min(1.0, max(0.0, (metrics['psnr'] - 10) / 40))
        
        # SSIM: already in [0, 1]
        normalized['ssim'] = metrics['ssim']
        
        # FFT: normalize based on expected range
        normalized['fft'] = min(1.0, max(0.0, metrics['fft']))
        
        # Edge: normalize based on expected range
        normalized['edge'] = min(1.0, max(0.0, metrics['edge']))
        
        # Texture: normalize based on expected range
        normalized['texture'] = min(1.0, max(0.0, metrics['texture']))
        
        # Noise reduction: typically [0, 1]
        normalized['noise_reduction'] = metrics['noise_reduction']
        
        # Artifact: lower is better, so invert
        normalized['artifact'] = 1.0 - min(1.0, max(0.0, metrics['artifact']))
        
        return normalized
    
    def _calculate_fft_score(self, image: np.ndarray, signatures: Dict) -> float:
        """Calculate FFT-based quality score."""
        if 'psd' not in signatures:
            return 0.5  # Default score if no FFT analysis available
        
        psd_info = signatures['psd']
        
        # Good denoising should have:
        # 1. Reasonable frequency distribution
        # 2. Not too much high-frequency noise
        # 3. Preserved low-frequency content
        
        freq_ratio = psd_info['freq_ratio']
        spectral_centroid = psd_info['spectral_centroid']
        
        # Ideal frequency ratio for natural images (empirically determined)
        ideal_freq_ratio = 0.3
        freq_score = np.exp(-abs(freq_ratio - ideal_freq_ratio) / 0.2)
        
        # Spectral centroid should be balanced
        normalized_centroid = spectral_centroid / len(psd_info['radial_profile'])
        centroid_score = 1.0 - abs(normalized_centroid - 0.3)
        
        return (freq_score + centroid_score) / 2
    
    def _calculate_edge_score(self, original: np.ndarray, 
                            denoised: np.ndarray) -> float:
        """Calculate edge preservation score."""
        import cv2
        
        # Compute edges
        orig_edges = cv2.Canny((original * 255).astype(np.uint8), 50, 150)
        den_edges = cv2.Canny((denoised * 255).astype(np.uint8), 50, 150)
        
        # Calculate overlap
        orig_edges_norm = orig_edges.astype(np.float64) / 255.0
        den_edges_norm = den_edges.astype(np.float64) / 255.0
        
        # True positive rate
        true_positives = np.sum(orig_edges_norm * den_edges_norm)
        total_original_edges = np.sum(orig_edges_norm)
        
        if total_original_edges > 0:
            edge_preservation = true_positives / total_original_edges
        else:
            edge_preservation = 1.0
        
        # False positive penalty
        false_positives = np.sum(den_edges_norm * (1 - orig_edges_norm))
        total_denoised_edges = np.sum(den_edges_norm)
        
        if total_denoised_edges > 0:
            false_positive_rate = false_positives / total_denoised_edges
        else:
            false_positive_rate = 0.0
        
        # Combined score
        edge_score = edge_preservation - 0.5 * false_positive_rate
        return max(0.0, min(1.0, edge_score))
    
    def _estimate_edge_score_no_reference(self, denoised: np.ndarray) -> float:
        """Estimate edge quality without reference image."""
        import cv2
        
        # Calculate edge strength and coherence
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge coherence (how well-defined edges are)
        mean_grad = np.mean(grad_magnitude)
        std_grad = np.std(grad_magnitude)
        
        # Good denoising should have clear, distinct edges
        edge_coherence = std_grad / (mean_grad + 1e-8)
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, edge_coherence / 2.0))
    
    def _calculate_texture_score(self, image: np.ndarray, signatures: Dict) -> float:
        """Calculate texture preservation score."""
        if 'texture' not in signatures:
            return 0.5  # Default score
        
        texture_info = signatures['texture']
        
        # Good texture should have:
        # 1. Appropriate entropy (not too smooth, not too noisy)
        # 2. Reasonable contrast
        # 3. Balanced homogeneity
        
        entropy = texture_info['lbp_entropy']
        contrast = texture_info['glcm_contrast']
        homogeneity = texture_info['glcm_homogeneity']
        
        # Ideal entropy for natural images (empirically determined)
        ideal_entropy = 4.0
        entropy_score = np.exp(-abs(entropy - ideal_entropy) / 2.0)
        
        # Contrast should be moderate
        contrast_score = np.exp(-abs(contrast - 50) / 25)
        
        # Homogeneity should be balanced
        homogeneity_score = homogeneity  # Already normalized
        
        return (entropy_score + contrast_score + homogeneity_score) / 3
    
    def _calculate_noise_reduction_score(self, noisy: np.ndarray, 
                                       denoised: np.ndarray) -> float:
        """Calculate how effectively noise was reduced."""
        # Compare local variance reduction
        noisy_var = np.var(noisy)
        denoised_var = np.var(denoised)
        
        # Noise reduction ratio
        if noisy_var > 0:
            noise_reduction = 1.0 - (denoised_var / noisy_var)
        else:
            noise_reduction = 0.0
        
        return max(0.0, min(1.0, noise_reduction))
    
    def _calculate_artifact_score(self, image: np.ndarray) -> float:
        """Calculate artifact presence score (lower is better)."""
        # Look for common denoising artifacts:
        # 1. Over-smoothing (too low local variance)
        # 2. Ringing artifacts (oscillations near edges)
        # 3. Blocking artifacts
        
        # Local variance analysis
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
        
        # Over-smoothing indicator
        low_variance_ratio = np.sum(local_var < 0.001) / image.size
        
        # Ringing artifacts (high-frequency oscillations)
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Look for oscillatory patterns
        high_grad_ratio = np.sum(grad_magnitude > 0.1) / image.size
        
        # Combine artifact indicators
        artifact_score = 0.5 * low_variance_ratio + 0.3 * high_grad_ratio
        
        return min(1.0, artifact_score)
    
    def _calculate_smoothness_score(self, image: np.ndarray) -> float:
        """Calculate smoothness score for the image."""
        # Calculate total variation
        grad_x = np.diff(image, axis=1)
        grad_y = np.diff(image, axis=0)
        
        tv_x = np.sum(np.abs(grad_x))
        tv_y = np.sum(np.abs(grad_y))
        total_variation = tv_x + tv_y
        
        # Normalize by image size
        normalized_tv = total_variation / image.size
        
        # Convert to smoothness score (lower TV = higher smoothness)
        smoothness = np.exp(-normalized_tv * 10)
        
        return smoothness
    
    def _estimate_psnr_no_reference(self, noisy: np.ndarray, 
                                  denoised: np.ndarray) -> float:
        """Estimate PSNR without reference image."""
        # Use difference between noisy and denoised as noise estimate
        noise_estimate = noisy - denoised
        noise_power = np.mean(noise_estimate**2)
        
        # Signal power (estimate from denoised image)
        signal_power = np.mean(denoised**2)
        
        if noise_power > 0:
            snr = signal_power / noise_power
            psnr_estimate = 10 * np.log10(snr)
        else:
            psnr_estimate = 40  # High value if no noise detected
        
        return max(10, min(50, psnr_estimate))  # Clamp to reasonable range
    
    def _estimate_ssim_no_reference(self, image: np.ndarray) -> float:
        """Estimate SSIM without reference (based on local structure)."""
        # Calculate local structure similarity within the image
        # Use autocorrelation as a proxy for structural consistency
        
        # Split image into overlapping patches
        patch_size = 8
        stride = 4
        patches = []
        
        for i in range(0, image.shape[0] - patch_size, stride):
            for j in range(0, image.shape[1] - patch_size, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        
        if len(patches) < 2:
            return 0.5
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(min(len(patches), 100)):  # Limit for efficiency
            for j in range(i+1, min(len(patches), 100)):
                corr = np.corrcoef(patches[i], patches[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            mean_correlation = np.mean(correlations)
            return mean_correlation
        else:
            return 0.5
    
    def _analyze_score_distribution(self, scores: Dict) -> Dict:
        """Analyze the distribution of scores across methods."""
        score_values = list(scores.values())
        
        return {
            'mean_score': np.mean(score_values),
            'std_score': np.std(score_values),
            'max_score': np.max(score_values),
            'min_score': np.min(score_values),
            'score_range': np.max(score_values) - np.min(score_values)
        }