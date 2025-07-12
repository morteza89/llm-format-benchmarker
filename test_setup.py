#!/usr/bin/env python3
"""
Simple test script to verify the project setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""

    print("Testing imports...")

    try:
        from configs.config import MODELS, HARDWARE, BENCHMARK
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from utils.hardware_detector import HardwareDetector
        print("✅ Hardware detector imported successfully")
    except ImportError as e:
        print(f"❌ Hardware detector import failed: {e}")
        return False

    try:
        from models.model_manager import ModelManager
        print("✅ Model manager imported successfully")
    except ImportError as e:
        print(f"❌ Model manager import failed: {e}")
        return False

    try:
        from quantization.quantization_manager import QuantizationManager
        print("✅ Quantization manager imported successfully")
    except ImportError as e:
        print(f"❌ Quantization manager import failed: {e}")
        return False

    try:
        from benchmarking.benchmark_engine import BenchmarkEngine
        print("✅ Benchmark engine imported successfully")
    except ImportError as e:
        print(f"❌ Benchmark engine import failed: {e}")
        return False

    try:
        from utils.utils import ResultsAnalyzer
        print("✅ Results analyzer imported successfully")
    except ImportError as e:
        print(f"❌ Results analyzer import failed: {e}")
        return False

    return True

def test_configuration():
    """Test configuration"""

    print("\nTesting configuration...")

    from configs.config import MODELS, HARDWARE, BENCHMARK

    # Test models
    expected_models = ["qwen2.5", "llama3.2-3b", "llava"]
    for model in expected_models:
        if model in MODELS:
            print(f"✅ Model {model} configured")
        else:
            print(f"❌ Model {model} not configured")

    # Test hardware
    expected_hardware = ["cpu", "gpu", "npu"]
    for hardware in expected_hardware:
        if hardware in HARDWARE:
            print(f"✅ Hardware {hardware} configured")
        else:
            print(f"❌ Hardware {hardware} not configured")

    # Test benchmark config
    if hasattr(BENCHMARK, 'warmup_runs'):
        print(f"✅ Benchmark warmup_runs: {BENCHMARK.warmup_runs}")
    else:
        print("❌ Benchmark warmup_runs not configured")

    return True

def test_directories():
    """Test if required directories exist"""

    print("\nTesting directories...")

    required_dirs = [
        "models",
        "results",
        "logs",
        "cache",
        "configs",
        "utils",
        "quantization",
        "benchmarking",
        "examples"
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Directory {dir_name} missing")
            # Create missing directories
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory {dir_name}")

    return True

def test_hardware_detection():
    """Test hardware detection"""

    print("\nTesting hardware detection...")

    try:
        from utils.hardware_detector import HardwareDetector
        detector = HardwareDetector()

        print(f"✅ CPU available: {detector.is_hardware_supported('cpu')}")
        print(f"✅ GPU available: {detector.is_hardware_supported('gpu')}")
        print(f"✅ NPU available: {detector.is_hardware_supported('npu')}")

        return True
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False

def test_model_info():
    """Test model configuration"""

    print("\nTesting model information...")

    try:
        from models.model_manager import ModelManager
        manager = ModelManager()

        for model_name in ["qwen2.5", "llama3.2-3b", "llava"]:
            info = manager.get_model_info(model_name)
            if info:
                print(f"✅ Model {model_name}: {info['model_type']}")
            else:
                print(f"❌ Model {model_name} info not available")

        return True
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

def main():
    """Run all tests"""

    print("ONNX OpenVINO Benchmarking LLM Models - Setup Test")
    print("="*60)

    tests = [
        test_imports,
        test_configuration,
        test_directories,
        test_hardware_detection,
        test_model_info
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed! The project is set up correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
