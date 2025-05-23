"""
QuadNet Main Entry Point
Command-line interface for running QuadNet salt-and-pepper noise removal.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

from src.quadnet_pipeline import QuadNetPipeline
from src.utils import load_image, add_salt_pepper_noise, save_image, calculate_psnr
from config import CONFIG

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QuadNet: Four-Phase Salt-and-Pepper Noise Removal Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single noisy image
  python main.py --input noisy_image.png --output results/

  # Process with original image for evaluation
  python main.py --input noisy.png --original clean.png --output results/

  # Add noise to clean image and process
  python main.py --input clean.png --add-noise 0.1 --output results/

  # Batch processing
  python main.py --batch-input images/ --output batch_results/

  # Generate synthetic test data
  python main.py --generate-test-data 10 --output test_data/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                           help='Path to input noisy image')
    input_group.add_argument('--batch-input', type=str,
                           help='Directory containing noisy images for batch processing')
    input_group.add_argument('--generate-test-data', type=int, metavar='N',
                           help='Generate N synthetic test images with salt-pepper noise')
    
    # Output options
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    
    # Processing options
    parser.add_argument('--original', type=str,
                       help='Path to original clean image (for evaluation)')
    parser.add_argument('--add-noise', type=float, metavar='AMOUNT',
                       help='Add salt-pepper noise to input image (0.0-1.0)')
    parser.add_argument('--noise-ratio', type=float, default=0.5,
                       help='Salt vs pepper ratio when adding noise (default: 0.5)')
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file (JSON)')
    parser.add_argument('--max-size', type=int, default=1024,
                       help='Maximum image dimension (default: 1024)')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Do not save intermediate results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    # Analysis options
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed performance report')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarking tests')
    
    return parser.parse_args()

def load_custom_config(config_path: str):
    """Load custom configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update CONFIG object with custom values
        for key, value in config_dict.items():
            if hasattr(CONFIG, key):
                setattr(CONFIG, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        
        print(f"Loaded custom configuration from {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def generate_test_data(num_images: int, output_dir: str):
    """Generate synthetic test data with various noise levels."""
    print(f"🧪 Generating {num_images} synthetic test images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple test pattern
    def create_test_image(size=(256, 256)):
        """Create synthetic test image with various patterns."""
        image = np.zeros(size)
        
        # Add geometric patterns
        center = np.array(size) // 2
        y, x = np.ogrid[:size[0], :size[1]]
        
        # Circular pattern
        radius = min(size) // 4
        circle_mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        image[circle_mask] = 0.8
        
        # Add stripes
        image[::10, :] = 0.3
        image[:, ::15] = 0.6
        
        # Add some noise-like texture
        texture = np.random.normal(0, 0.05, size)
        image = np.clip(image + texture, 0, 1)
        
        return image
    
    noise_levels = np.linspace(0.05, 0.25, num_images)
    
    for i, noise_level in enumerate(noise_levels):
        # Generate clean image
        clean_image = create_test_image()
        
        # Add salt-pepper noise
        noisy_image = add_salt_pepper_noise(clean_image, amount=noise_level)
        
        # Save images
        image_name = f"test_{i+1:03d}_noise_{noise_level:.3f}"
        save_image(clean_image, os.path.join(output_dir, f"{image_name}_clean.png"))
        save_image(noisy_image, os.path.join(output_dir, f"{image_name}_noisy.png"))
        
        print(f"   Generated {image_name} (noise level: {noise_level:.3f})")
    
    print(f"✅ Test data saved to {output_dir}")

def run_benchmark():
    """Run benchmarking tests."""
    print("🏃 Running QuadNet benchmark tests...")
    
    # Create test image
    test_image = np.random.rand(512, 512) * 0.5 + 0.25
    
    # Test different noise levels
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    pipeline = QuadNetPipeline()
    
    benchmark_results = {}
    
    for noise_level in noise_levels:
        print(f"\n📊 Testing noise level: {noise_level}")
        
        # Add noise
        noisy = add_salt_pepper_noise(test_image, amount=noise_level)
        
        # Process
        result = pipeline.process_image(noisy, test_image, save_intermediate=False)
        
        benchmark_results[noise_level] = {
            'processing_time': result['processing_time']['total'],
            'best_method': result['final_result']['best_method'],
            'best_score': result['final_result']['best_score'],
            'psnr_improvement': result['phase4_evaluation']['detailed_metrics'][
                result['final_result']['best_method']
            ]['psnr'] - calculate_psnr(test_image, noisy)
        }
    
    # Print benchmark summary
    print("\n📈 Benchmark Results Summary:")
    print("=" * 60)
    print(f"{'Noise Level':<12} {'Time (s)':<10} {'Best Method':<15} {'PSNR Gain':<10}")
    print("-" * 60)
    
    for noise_level, stats in benchmark_results.items():
        print(f"{noise_level:<12.3f} {stats['processing_time']:<10.2f} "
              f"{stats['best_method']:<15} {stats['psnr_improvement']:<10.2f}")
    
    return benchmark_results

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load custom configuration if provided
    if args.config:
        load_custom_config(args.config)
    
    # Update configuration based on command line arguments
    if args.max_size:
        CONFIG.MAX_IMAGE_SIZE = args.max_size
    if args.no_intermediate:
        CONFIG.SAVE_INTERMEDIATE = False
    if args.quiet:
        CONFIG.VERBOSE = False
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Handle different modes
    if args.generate_test_data:
        generate_test_data(args.generate_test_data, args.output)
        return
    
    if args.benchmark:
        benchmark_results = run_benchmark()
        
        # Save benchmark results
        with open(os.path.join(args.output, 'benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        return
    
    # Initialize pipeline
    pipeline = QuadNetPipeline()
    
    # Single image processing
    if args.input:
        print(f"📷 Processing single image: {args.input}")
        
        # Load input image
        if not os.path.exists(args.input):
            print(f"Error: Input image not found: {args.input}")
            sys.exit(1)
        
        input_image = load_image(args.input)
        
        # Add noise if requested
        if args.add_noise:
            print(f"🧂 Adding salt-pepper noise (amount: {args.add_noise})")
            noisy_image = add_salt_pepper_noise(
                input_image, amount=args.add_noise, salt_vs_pepper=args.noise_ratio
            )
            # Save original as reference
            save_image(input_image, os.path.join(args.output, "original_clean.png"))
            save_image(noisy_image, os.path.join(args.output, "input_noisy.png"))
        else:
            noisy_image = input_image
        
        # Load original image if provided
        original_image = None
        if args.original:
            if os.path.exists(args.original):
                original_image = load_image(args.original)
            else:
                print(f"Warning: Original image not found: {args.original}")
        elif args.add_noise:
            original_image = input_image  # Use clean version as original
        
        # Process image
        result = pipeline.process_image(
            noisy_image, 
            original_image, 
            save_intermediate=CONFIG.SAVE_INTERMEDIATE,
            output_dir=args.output
        )
        
        # Generate report if requested
        if args.report:
            report = pipeline.get_performance_report()
            
            # Save report as JSON
            with open(os.path.join(args.output, 'performance_report.json'), 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Print summary
            print("\n📊 Performance Report Summary:")
            print("=" * 50)
            print(f"Best Method: {report['best_result']['method']}")
            print(f"Overall Score: {report['best_result']['score']:.4f}")
            print(f"Processing Time: {report['processing_efficiency']['total_time']:.2f}s")
            
            if 'improvement_metrics' in report['best_result'] and report['best_result']['improvement_metrics']:
                improvements = report['best_result']['improvement_metrics']
                if 'psnr_improvement_db' in improvements:
                    print(f"PSNR Improvement: {improvements['psnr_improvement_db']:.2f} dB")
                if 'ssim_improvement' in improvements:
                    print(f"SSIM Improvement: {improvements['ssim_improvement']:.3f}")
    
    # Batch processing
    elif args.batch_input:
        print(f"📁 Processing batch from directory: {args.batch_input}")
        
        if not os.path.isdir(args.batch_input):
            print(f"Error: Batch input directory not found: {args.batch_input}")
            sys.exit(1)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.batch_input).glob(f'*{ext}'))
            image_paths.extend(Path(args.batch_input).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"Error: No image files found in {args.batch_input}")
            sys.exit(1)
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process batch
        batch_results = pipeline.process_batch(image_paths, args.output)
        
        # Save batch results
        with open(os.path.join(args.output, 'batch_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for path, result in batch_results['results'].items():
                if 'error' not in result:
                    serializable_results[path] = {
                        'best_method': result['final_result']['best_method'],
                        'best_score': result['final_result']['best_score'],
                        'processing_time': result['processing_time']['total'],
                        'all_scores': result['final_result']['all_scores']
                    }
                else:
                    serializable_results[path] = result
            
            json.dump({
                'results': serializable_results,
                'statistics': batch_results['statistics']
            }, f, indent=2, default=str)
        
        print(f"\n✅ Batch processing complete!")
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()