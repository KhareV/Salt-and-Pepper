"""
Batch Processing Example for QuadNet
Process multiple images and generate comparison reports.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadnet_pipeline import QuadNetPipeline
from src.utils import load_image, add_salt_pepper_noise, save_image

def process_folder(input_folder, output_folder, add_noise_level=None):
    """Process all images in a folder."""
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        for file in os.listdir(input_folder):
            if file.lower().endswith(ext):
                image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Initialize pipeline
    pipeline = QuadNetPipeline()
    
    # Process each image
    results = []
    
    for i, filename in enumerate(image_files):
        print(f"\n📷 Processing {i+1}/{len(image_files)}: {filename}")
        
        try:
            # Load image
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)
            
            # Create output directory for this image
            name_without_ext = os.path.splitext(filename)[0]
            image_output_dir = os.path.join(output_folder, name_without_ext)
            
            # Add noise if requested
            if add_noise_level:
                original_image = image.copy()
                noisy_image = add_salt_pepper_noise(image, amount=add_noise_level)
                save_image(original_image, os.path.join(image_output_dir, "original.png"))
                save_image(noisy_image, os.path.join(image_output_dir, "noisy.png"))
            else:
                noisy_image = image
                original_image = None
            
            # Process with QuadNet
            result = pipeline.process_image(
                noisy_image,
                original_image=original_image,
                save_intermediate=True,
                output_dir=image_output_dir
            )
            
            results.append({
                'filename': filename,
                'best_method': result['final_result']['best_method'],
                'best_score': result['final_result']['best_score'],
                'processing_time': result['processing_time']['total']
            })
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            results.append({
                'filename': filename,
                'error': str(e)
            })
    
    # Generate summary report
    print("\n📊 Generating summary report...")
    
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        # Summary statistics
        avg_score = np.mean([r['best_score'] for r in successful_results])
        avg_time = np.mean([r['processing_time'] for r in successful_results])
        
        # Method frequency
        methods = [r['best_method'] for r in successful_results]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Create summary plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Method distribution
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        
        axes[0].bar(methods, counts, color='skyblue', alpha=0.7)
        axes[0].set_title('Best Method Distribution')
        axes[0].set_ylabel('Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Score distribution
        scores = [r['best_score'] for r in successful_results]
        axes[1].hist(scores, bins=10, color='lightgreen', alpha=0.7)
        axes[1].set_title('Score Distribution')
        axes[1].set_xlabel('QuadNet Score')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'batch_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n✅ Batch Processing Summary:")
        print("=" * 50)
        print(f"Total images processed: {len(successful_results)}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Most successful method: {max(method_counts.keys(), key=method_counts.get)}")
        
        print(f"\n📁 Results saved to: {output_folder}")

if __name__ == "__main__":
    # Example usage
    input_folder = "test_images"
    output_folder = "batch_output"
    
    # Process with noise addition
    process_folder(input_folder, output_folder, add_noise_level=0.1)