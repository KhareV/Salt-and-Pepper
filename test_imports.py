"""Test all imports"""
try:
    print("Testing imports...")
    
    from src.phase1_detection import NoiseDetector
    print("✅ Phase 1 import OK")
    
    from src.phase2_denoisers import DenoisingEnsemble
    print("✅ Phase 2 import OK")
    
    from src.phase3_analysis import StatisticalAnalyzer
    print("✅ Phase 3 import OK")
    
    from src.phase4_scoring import PerformanceScorer
    print("✅ Phase 4 import OK")
    
    from src.quadnet_pipeline import QuadNetPipeline
    print("✅ Pipeline import OK")
    
    from src.utils import load_image, calculate_psnr, calculate_ssim
    print("✅ Utils import OK")
    
    print("\n🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")