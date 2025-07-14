#!/usr/bin/env python3
"""
Test script for Intel NPU using OpenVINO
"""

import os
import time
import psutil
from pathlib import Path


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_intel_npu():
    print("=== Intel NPU Test ===")

    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")

        # Check available devices
        core = ov.Core()
        devices = core.available_devices
        print(f"Available OpenVINO devices: {devices}")

        # Look for NPU device
        npu_device = None
        for device in devices:
            if 'NPU' in device:
                npu_device = device
                break

        if npu_device:
            print(f"✅ Intel NPU detected: {npu_device}")

            # Get NPU properties
            try:
                device_name = core.get_property(npu_device, "FULL_DEVICE_NAME")
                print(f"NPU Device Name: {device_name}")
            except Exception as e:
                print(f"Could not get NPU properties: {e}")
        else:
            print("❌ Intel NPU not found in OpenVINO devices")

        # Test CPU as fallback
        cpu_device = 'CPU'
        if cpu_device in devices:
            print(f"✅ CPU device available: {cpu_device}")
            device_name = core.get_property(cpu_device, "FULL_DEVICE_NAME")
            print(f"CPU Device Name: {device_name}")

        return npu_device, devices

    except ImportError as e:
        print(f"❌ OpenVINO not available: {e}")
        return None, []
    except Exception as e:
        print(f"❌ OpenVINO initialization failed: {e}")
        return None, []


def test_openvino_genai():
    print("\n=== OpenVINO GenAI Test ===")

    try:
        import openvino_genai as ov_genai
        print(f"✅ OpenVINO GenAI available")

        # Test simple text generation
        print("Testing OpenVINO GenAI capabilities...")

    except ImportError as e:
        print(f"❌ OpenVINO GenAI not available: {e}")
    except Exception as e:
        print(f"❌ OpenVINO GenAI test failed: {e}")


if __name__ == "__main__":
    npu_device, devices = test_intel_npu()
    test_openvino_genai()

    print(f"\n=== Summary ===")
    print(f"OpenVINO devices: {devices}")
    print(f"NPU device: {npu_device if npu_device else 'Not found'}")
    print(f"Memory usage: {get_memory_usage():.1f} MB")
