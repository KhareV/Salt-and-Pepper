"""
Performance Test Script for QuadNet
Tests different configurations and measures performance.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadnet_pipeline import QuadNetPipeline
from src.utils import add_salt_pepper_noise
from config import CONFIG

def performance_test():
    """Run comprehensive performance tests."""
    
    print("🏃‍♂️ Running QuadNet Performance Tests...")
    
    # Test configurations
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    # Initialize pipeline
    pipeline = QuadNetPipeline()
    
    results = {}
    
    for size in test_sizes:
        print(f"\n📏 Testing image size: {size[0]}x{size[1]}")
        
        # Create test image
        test_image = np.random.rand(*size) * 0.6 + 0.2
        
        size_results = {}
        
        for noise_level in noise_levels:
            print(f"   🧂 Noise level: {noise_level}")
            
            # Add noise
            noisy_image = add_salt_pepper_noise(test_image, amount=noise_level)
            
            # Measure processing time
            start_time = time.time()
            
            result = pipeline.process_image(
                noisy_image, 
                test_image, 
                save_intermediate=False
            )
            
            processing_time = time.time() - start_time
            
            size_results[noise_level] = {
                'processing_time': processing_time,
                'best_score': result['final_result']['best_score'],
                'best_method': result['final_result']['best_method']
            }
            
            print(f"      ⏱️ Time: {processing_time:.2f}s, Score: {result['final_result']['best_score']:.3f}")
        
        results[size] = size_results
    
    # Create performance visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Processing time vs image size
    sizes_str = [f"{s[0]}x{s[1]}" for s in test_sizes]
    avg_times = []
    
    for size in test_sizes:
        times = [results[size][nl]['processing_time'] for nl in noise_levels]
        avg_times.append(np.mean(times))
    
    axes[0, 0].bar(sizes_str, avg_times, color='blue', alpha=0.7)
    axes[0, 0].set_title('Average Processing Time vs Image Size')
    axes[0, 0].set_ylabel('Time (seconds)')
    
    # Plot 2: Score vs noise level
    for size in test_sizes:
        scores = [results[size][nl]['best_score'] for nl in noise_levels]
        axes[0, 1].plot(noise_levels, scores, 'o-', label=f"{size[0]}x{size[1]}")
    
    axes[0, 1].set_title('Performance vs Noise Level')
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('QuadNet Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time vs noise level
    for size in test_sizes:
        times = [results[size][nl]['processing_time'] for nl in noise_levels]
        axes[1, 0].plot(noise_levels, times, 's-', label=f"{size[0]}x{size[1]}")
    
    axes[1, 0].set_title('Processing Time vs Noise Level')
    axes[1, 0].set_xlabel('Noise Level')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Method distribution
    all_methods = []
    for size in test_sizes:
        for nl in noise_levels:
            all_methods.append(results[size][nl]['best_method'])
    
    method_counts = {}
    for method in all_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    axes[1, 1].pie(counts, labels=methods, autopct='%1.1f%%')
    axes[1, 1].set_title('Best Method Distribution')
    
    plt.tight_layout()
    plt.savefig('performance_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Performance test complete!")
    print(f"📊 Results visualization saved as 'performance_test_results.png'")

if __name__ == "__main__":
    performance_test()