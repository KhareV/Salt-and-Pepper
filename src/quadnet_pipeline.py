"""
QuadNet Main Pipeline
Orchestrates the four phases of the QuadNet framework.
"""
import cv2
import numpy as np
import time
from typing import Dict, Optional, Tuple
import os

from src.phase1_detection import NoiseDetector
from src.phase2_denoisers import DenoisingEnsemble
from src.phase3_analysis import StatisticalAnalyzer
from src.phase4_scoring import PerformanceScorer
from src.utils import *
from config import CONFIG

class QuadNetPipeline:
    """Main QuadNet pipeline orchestrating all four phases."""
    
    def __init__(self, config=None):
        """Initialize QuadNet pipeline with optional custom configuration."""
        self.config = config or CONFIG
        
        # Initialize phase modules
        self.noise_detector = NoiseDetector()
        self.denoising_ensemble = DenoisingEnsemble()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.performance_scorer = PerformanceScorer()
        
        # Pipeline state
        self.last_results = None
        self.processing_time = {}
    
    def process_image(self, noisy_image: np.ndarray, 
                     original_image: Optional[np.ndarray] = None,
                     save_intermediate: bool = None,
                     output_dir: str = None) -> Dict:
        """
        Process image through complete QuadNet pipeline.
        
        Args:
            noisy_image: Input noisy image (0-1 range)
            original_image: Original clean image for evaluation (optional)
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save results
            
        Returns:
            Complete processing results
        """
        if save_intermediate is None:
            save_intermediate = self.config.SAVE_INTERMEDIATE
        
        if output_dir is None:
            output_dir = "output"
        
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        if self.config.VERBOSE:
            print("🚀 Starting QuadNet Processing...")
            print(f"   Input image shape: {noisy_image.shape}")
        
        # Resize if necessary
        original_shape = noisy_image.shape
        noisy_image = resize_image_if_needed(noisy_image, self.config.MAX_IMAGE_SIZE)
        if original_image is not None:
            original_image = resize_image_if_needed(original_image, self.config.MAX_IMAGE_SIZE)
        
        results = {
            'input_image': noisy_image,
            'original_image': original_image,
            'original_shape': original_shape,
            'processing_metadata': {
                'timestamp': time.time(),
                'config': self.config.__dict__
            }
        }
        
        # Phase 1: Noise Detection and Characterization
        phase1_start = time.time()
        if self.config.VERBOSE:
            print("🔍 Phase 1: Noise Detection and Characterization")
        
        noise_info = self.noise_detector.detect_salt_pepper_noise(noisy_image)
        results['phase1_noise_analysis'] = noise_info
        
        self.processing_time['phase1'] = time.time() - phase1_start
        
        if self.config.VERBOSE:
            print(f"   ✓ Noise density: {noise_info['noise_density']:.3f}")
            print(f"   ✓ Salt ratio: {noise_info['salt_ratio']:.3f}")
            print(f"   ✓ Pepper ratio: {noise_info['pepper_ratio']:.3f}")
        
        if save_intermediate:
            save_image(noise_info['noise_mask'].astype(np.float64), 
                      os.path.join(output_dir, "noise_mask.png"))
        
        # Phase 2: Apply Ensemble of Denoisers
        phase2_start = time.time()
        if self.config.VERBOSE:
            print("🛠️  Phase 2: Ensemble Denoising")
        
        denoised_images = self.denoising_ensemble.apply_all_denoisers(
            noisy_image, noise_info
        )
        results['phase2_denoised_images'] = denoised_images
        
        self.processing_time['phase2'] = time.time() - phase2_start
        
        if self.config.VERBOSE:
            print(f"   ✓ Applied {len(denoised_images)} denoising methods")
        
        if save_intermediate:
            for method_name, denoised_img in denoised_images.items():
                save_image(denoised_img, 
                          os.path.join(output_dir, f"denoised_{method_name}.png"))
        
        # Phase 3: Statistical Signature Extraction
        phase3_start = time.time()
        if self.config.VERBOSE:
            print("📊 Phase 3: Statistical Analysis")
        
        statistical_signatures = {}
        for method_name, denoised_img in denoised_images.items():
            signatures = self.statistical_analyzer.extract_signatures(
                original_image, denoised_img
            )
            statistical_signatures[method_name] = signatures
        
        results['phase3_statistical_signatures'] = statistical_signatures
        self.processing_time['phase3'] = time.time() - phase3_start
        
        if self.config.VERBOSE:
            print(f"   ✓ Extracted statistical signatures for all methods")
        
        # Phase 4: Performance Scoring and Selection
        phase4_start = time.time()
        if self.config.VERBOSE:
            print("🎯 Phase 4: Performance Scoring and Selection")
        
        evaluation_results = self.performance_scorer.evaluate_all_methods(
            original_image, noisy_image, denoised_images, statistical_signatures
        )
        results['phase4_evaluation'] = evaluation_results
        
        self.processing_time['phase4'] = time.time() - phase4_start
        
        # Final results compilation
        best_method = evaluation_results['best_method']
        best_image = denoised_images[best_method]
        best_score = evaluation_results['scores'][best_method]
        
        results['final_result'] = {
            'best_method': best_method,
            'best_image': best_image,
            'best_score': best_score,
            'all_scores': evaluation_results['scores'],
            'ranking': evaluation_results['ranking']
        }
        
        total_time = time.time() - start_time
        self.processing_time['total'] = total_time
        results['processing_time'] = self.processing_time.copy()
        
        if self.config.VERBOSE:
            print(f"✅ QuadNet Processing Complete!")
            print(f"   🏆 Best method: {best_method}")
            print(f"   📈 Best score: {best_score:.4f}")
            print(f"   ⏱️  Total time: {total_time:.2f} seconds")
            
            # Show top 3 methods
            print("   🥇 Top 3 methods:")
            for i, (method, score) in enumerate(evaluation_results['ranking'][:3]):
                print(f"      {i+1}. {method}: {score:.4f}")
        
        # Save final results
        if save_intermediate:
            save_image(best_image, os.path.join(output_dir, "final_result.png"))
            
            # Create comparison plot
            create_comparison_plot(
                original_image if original_image is not None else noisy_image,
                noisy_image,
                {
                    'denoised_images': denoised_images,
                    'scores': evaluation_results['scores'],
                    'best_method': best_method
                },
                os.path.join(output_dir, "comparison_plot.png")
            )
        
        self.last_results = results
        return results
    
    def process_batch(self, image_paths: list, 
                     output_base_dir: str = "batch_output",
                     original_paths: list = None) -> Dict:
        """
        Process multiple images in batch mode.
        
        Args:
            image_paths: List of paths to noisy images
            output_base_dir: Base directory for outputs
            original_paths: List of paths to original images (optional)
            
        Returns:
            Batch processing results
        """
        batch_results = {}
        batch_stats = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'avg_time_per_image': 0,
            'method_performance': {}
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            if self.config.VERBOSE:
                print(f"\n📷 Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Load images
                noisy_image = load_image(image_path)
                original_image = None
                
                if original_paths and i < len(original_paths):
                    original_image = load_image(original_paths[i])
                
                # Create output directory for this image
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.join(output_base_dir, f"image_{i+1:03d}_{image_name}")
                
                # Process image
                result = self.process_image(
                    noisy_image, original_image, 
                    save_intermediate=True, 
                    output_dir=output_dir
                )
                
                batch_results[image_path] = result
                batch_stats['successful'] += 1
                
                # Update method performance statistics
                for method, score in result['phase4_evaluation']['scores'].items():
                    if method not in batch_stats['method_performance']:
                        batch_stats['method_performance'][method] = []
                    batch_stats['method_performance'][method].append(score)
                
            except Exception as e:
                if self.config.VERBOSE:
                    print(f"   ❌ Error processing {image_path}: {str(e)}")
                batch_results[image_path] = {'error': str(e)}
                batch_stats['failed'] += 1
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        batch_stats['total_time'] = total_time
        batch_stats['avg_time_per_image'] = total_time / len(image_paths)
        
        # Calculate average performance per method
        method_avg_performance = {}
        for method, scores in batch_stats['method_performance'].items():
            method_avg_performance[method] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        batch_stats['method_avg_performance'] = method_avg_performance
        
        # Find best overall method
        if method_avg_performance:
            best_overall_method = max(method_avg_performance.keys(), 
                                    key=lambda k: method_avg_performance[k]['mean_score'])
            batch_stats['best_overall_method'] = best_overall_method
        
        if self.config.VERBOSE:
            print(f"\n📊 Batch Processing Summary:")
            print(f"   ✅ Successfully processed: {batch_stats['successful']}")
            print(f"   ❌ Failed: {batch_stats['failed']}")
            print(f"   ⏱️  Total time: {total_time:.2f} seconds")
            print(f"   📈 Average time per image: {batch_stats['avg_time_per_image']:.2f} seconds")
            
            if 'best_overall_method' in batch_stats:
                print(f"   🏆 Best overall method: {batch_stats['best_overall_method']}")
        
        return {
            'results': batch_results,
            'statistics': batch_stats
        }
    
    def get_performance_report(self) -> Dict:
        """Generate detailed performance report from last processing."""
        if self.last_results is None:
            return {"error": "No processing results available"}
        
        results = self.last_results
        noise_info = results['phase1_noise_analysis']
        evaluation = results['phase4_evaluation']
        
        report = {
            'input_analysis': {
                'noise_density': noise_info['noise_density'],
                'salt_pepper_ratio': f"{noise_info['salt_ratio']:.2f}:{noise_info['pepper_ratio']:.2f}",
                'spatial_uniformity': noise_info['spatial_distribution']['uniformity_score'],
                'clustering_score': noise_info['clustering_info']['clustering_ratio']
            },
            'method_performance': {},
            'best_result': {
                'method': results['final_result']['best_method'],
                'score': results['final_result']['best_score'],
                'improvement_metrics': {}
            },
            'processing_efficiency': {
                'total_time': self.processing_time['total'],
                'phase_breakdown': {
                    'detection': self.processing_time['phase1'],
                    'denoising': self.processing_time['phase2'],
                    'analysis': self.processing_time['phase3'],
                    'scoring': self.processing_time['phase4']
                }
            }
        }
        
        # Detailed method performance
        for method, metrics in evaluation['detailed_metrics'].items():
            report['method_performance'][method] = {
                'overall_score': evaluation['scores'][method],
                'psnr': metrics.get('psnr', 'N/A'),
                'ssim': metrics.get('ssim', 'N/A'),
                'edge_preservation': metrics['edge'],
                'texture_score': metrics['texture'],
                'noise_reduction': metrics['noise_reduction']
            }
        
        # Calculate improvements if original is available
        if results['original_image'] is not None:
            original = results['original_image']
            noisy = results['input_image']
            best_denoised = results['final_result']['best_image']
            
            # PSNR improvement
            noisy_psnr = calculate_psnr(original, noisy)
            denoised_psnr = calculate_psnr(original, best_denoised)
            psnr_improvement = denoised_psnr - noisy_psnr
            
            # SSIM improvement
            noisy_ssim = calculate_ssim(original, noisy)
            denoised_ssim = calculate_ssim(original, best_denoised)
            ssim_improvement = denoised_ssim - noisy_ssim
            
            report['best_result']['improvement_metrics'] = {
                'psnr_improvement_db': psnr_improvement,
                'ssim_improvement': ssim_improvement,
                'relative_psnr_gain': (psnr_improvement / noisy_psnr) * 100 if noisy_psnr > 0 else 0
            }
        
        return report