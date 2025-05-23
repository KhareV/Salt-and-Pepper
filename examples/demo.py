
### 11. Example Script

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadnet_pipeline import QuadNetPipeline
from src.utils import add_salt_pepper_noise, create_comparison_plot
from config import CONFIG

def create_demo_image():
    """Create a synthetic demo image for testing."""
    # Create a test image with various features
    img = np.zeros((256, 256))
    
    # Add circles
    center1 = (64, 64)
    center2 = (192, 192)
    y, x = np.ogrid[:256, :256]
    
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 <= 30**2
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 <= 25**2
    
    img[mask1] = 0.8
    img[mask2] = 0.6
    
    # Add rectangles
    img[100:150, 180:230] = 0.9
    img[50:80, 150:200] = 0.4
    
    # Add diagonal lines
    for i in range(256):
        if i < 256:
            img[i, min(i, 255)] = 0.7
            img[i, max(0, 255-i)] = 0.3
    
    # Add some texture
    texture = np.random.normal(0, 0.02, (256, 256))
    img = np.clip(img + texture, 0, 1)
    
    return img

def run_demo():
    """Run complete QuadNet demonstration."""
    print("🚀 QuadNet Demo Starting...")
    
    # Create output directory
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create demo image
    print("📷 Creating demo image...")
    clean_image = create_demo_image()
    
    # Test different noise levels
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    # Initialize pipeline
    pipeline = QuadNetPipeline()
    
    results_summary = {}
    
    for noise_level in noise_levels:
        print(f"\n🧂 Testing noise level: {noise_level}")
        
        # Add noise
        noisy_image = add_salt_pepper_noise(clean_image, amount=noise_level)
        
        # Create subdirectory for this noise level
        level_output = os.path.join(output_dir, f"noise_{noise_level:.2f}")
        
        # Process with QuadNet
        result = pipeline.process_image(
            noisy_image,
            original_image=clean_image,
            save_intermediate=True,
            output_dir=level_output
        )
        
        # Store results
        results_summary[noise_level] = {
            'best_method': result['final_result']['best_method'],
            'best_score': result['final_result']['best_score'],
            'processing_time': result['processing_time']['total'],
            'psnr_improvement': result['phase4_evaluation']['detailed_metrics'][
                result['final_result']['best_method']
            ]['psnr'] - pipeline.performance_scorer._calculate_individual_metrics(
                clean_image, noisy_image, noisy_image, {}
            )['psnr']
        }
        
        # Create detailed comparison plot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Top row: Original, Noisy, Best Result, Noise Map
        axes[0, 0].imshow(clean_image, cmap='gray')
        axes[0, 0].set_title('Original Clean')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(noisy_image, cmap='gray')
        axes[0, 1].set_title(f'Noisy ({noise_level*100:.0f}%)')
        axes[0, 1].axis('off')
        
        best_image = result['final_result']['best_image']
        axes[0, 2].imshow(best_image, cmap='gray')
        axes[0, 2].set_title(f'QuadNet Result\n({result["final_result"]["best_method"]})')
        axes[0, 2].axis('off')
        
        noise_map = result['phase1_noise_analysis']['noise_mask']
        axes[0, 3].imshow(noise_map, cmap='Reds')
        axes[0, 3].set_title('Detected Noise Map')
        axes[0, 3].axis('off')
        
        # Bottom row: Top 3 individual methods
        top_methods = result['phase4_evaluation']['ranking'][:4]
        for i, (method, score) in enumerate(top_methods):
            if i < 4:
                method_image = result['phase2_denoised_images'][method]
                axes[1, i].imshow(method_image, cmap='gray')
                axes[1, i].set_title(f'{method}\nScore: {score:.3f}')
                axes[1, i].axis('off')
        
        plt.suptitle(f'QuadNet Results - Noise Level {noise_level*100:.0f}%', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(level_output, 'detailed_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary plot
    print("\n📊 Creating summary analysis...")
    
    noise_levels_list = list(results_summary.keys())
    best_methods = [results_summary[nl]['best_method'] for nl in noise_levels_list]
    best_scores = [results_summary[nl]['best_score'] for nl in noise_levels_list]
    processing_times = [results_summary[nl]['processing_time'] for nl in noise_levels_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Best scores vs noise level
    axes[0].plot(noise_levels_list, best_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Best Score')
    axes[0].set_title('QuadNet Performance vs Noise Level')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Processing time vs noise level
    axes[1].plot(noise_levels_list, processing_times, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Processing Time (s)')
    axes[1].set_title('Computational Efficiency')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Best method distribution
    method_counts = {}
    for method in best_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    axes[2].bar(methods, counts, color='green', alpha=0.7)
    axes[2].set_xlabel('Method')
    axes[2].set_ylabel('Times Selected as Best')
    axes[2].set_title('Best Method Distribution')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n✅ Demo Complete! Summary:")
    print("=" * 60)
    print(f"{'Noise Level':<12} {'Best Method':<15} {'Score':<8} {'Time (s)':<8}")
    print("-" * 60)
    
    for noise_level in noise_levels_list:
        stats = results_summary[noise_level]
        print(f"{noise_level:<12.2f} {stats['best_method']:<15} "
              f"{stats['best_score']:<8.3f} {stats['processing_time']:<8.2f}")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("🎉 QuadNet demonstration complete!")

if __name__ == "__main__":
    run_demo()