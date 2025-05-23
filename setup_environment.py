"""
Environment Setup Script for QuadNet
Automatically fixes NumPy compatibility issues and installs required dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} may not be compatible")
        print("Recommended: Python 3.7 or higher")
        return False

def setup_environment():
    """Set up the environment for QuadNet."""
    print("🚀 Setting up QuadNet environment...")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("Consider upgrading Python to 3.7 or higher")
    
    # Uninstall problematic packages first
    print("\n📦 Removing potentially problematic packages...")
    packages_to_remove = ["numpy", "matplotlib", "scikit-image"]
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Removing {package}")
    
    # Install compatible versions
    print("\n📦 Installing compatible package versions...")
    
    # Core packages with specific versions known to work together
    core_packages = [
        "numpy==1.24.3",
        "matplotlib==3.7.2", 
        "scikit-image==0.21.0",
        "scipy==1.10.1",
        "Pillow==9.5.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Failed to install {package}")
            return False
    
    # Additional packages
    additional_packages = [
        "tqdm>=4.62.0",
        "opencv-python>=4.5.0"
    ]
    
    for package in additional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    print("\n🧪 Testing installation...")
    
    # Test imports
    test_imports = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ("skimage", "import skimage; print(f'scikit-image {skimage.__version__}')"),
    ]
    
    all_good = True
    for name, test_code in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: Failed to import")
                all_good = False
        except Exception as e:
            print(f"❌ {name}: Error testing import - {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 Environment setup completed successfully!")
        print("You can now run QuadNet with:")
        print("  python main.py --input your_image.png --output results/")
    else:
        print("\n⚠️  Some packages may still have issues.")
        print("Try running the setup again or install packages manually.")
    
    return all_good

if __name__ == "__main__":
    setup_environment()
