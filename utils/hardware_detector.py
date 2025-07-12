"""
Hardware detection and capability checking utilities
"""

import os
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
import psutil
import cpuinfo

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detects available hardware and their capabilities"""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.cpu_info = self._get_cpu_info()
        self.gpu_info = self._get_gpu_info()
        self.npu_info = self._get_npu_info()

    def _get_system_info(self) -> Dict:
        """Get basic system information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }

    def _get_cpu_info(self) -> Dict:
        """Get CPU information"""
        cpu_info_data = cpuinfo.get_cpu_info()
        return {
            "brand": cpu_info_data.get("brand_raw", "Unknown"),
            "arch": cpu_info_data.get("arch", "Unknown"),
            "bits": cpu_info_data.get("bits", 64),
            "count": cpu_info_data.get("count", psutil.cpu_count()),
            "hz_actual": cpu_info_data.get("hz_actual_friendly", "Unknown"),
            "vendor_id": cpu_info_data.get("vendor_id_raw", "Unknown"),
            "flags": cpu_info_data.get("flags", [])
        }

    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        gpu_info = {
            "available": False,
            "devices": [],
            "intel_gpu": False,
            "nvidia_gpu": False,
            "total_memory_gb": 0
        }

        # Check for NVIDIA GPUs
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpu_info["devices"].append({
                        "id": i,
                        "name": name,
                        "memory_gb": round(memory_info.total / (1024**3), 2),
                        "vendor": "NVIDIA"
                    })

                    gpu_info["total_memory_gb"] += memory_info.total / (1024**3)
                    gpu_info["nvidia_gpu"] = True
                    gpu_info["available"] = True

            except Exception as e:
                logger.warning(f"Could not get NVIDIA GPU info: {e}")

        # Check for Intel GPUs (basic detection)
        if platform.system() == "Windows":
            try:
                # Check for Intel GPU through device manager or registry
                result = subprocess.run(
                    ["powershell", "-Command", "Get-WmiObject -Class Win32_VideoController | Select-Object Name"],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.lower()
                    if "intel" in output:
                        gpu_info["intel_gpu"] = True
                        gpu_info["available"] = True
                        # Add basic Intel GPU info
                        gpu_info["devices"].append({
                            "id": len(gpu_info["devices"]),
                            "name": "Intel Integrated Graphics",
                            "memory_gb": 0,  # Shared memory
                            "vendor": "Intel"
                        })

            except Exception as e:
                logger.warning(f"Could not detect Intel GPU: {e}")

        return gpu_info

    def _get_npu_info(self) -> Dict:
        """Get NPU information (Intel NPU detection)"""
        npu_info = {
            "available": False,
            "devices": [],
            "intel_npu": False
        }

        # Check for Intel NPU through OpenVINO
        if OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                available_devices = core.available_devices

                for device in available_devices:
                    if "NPU" in device.upper():
                        npu_info["available"] = True
                        npu_info["intel_npu"] = True
                        npu_info["devices"].append({
                            "id": len(npu_info["devices"]),
                            "name": device,
                            "vendor": "Intel"
                        })

            except Exception as e:
                logger.warning(f"Could not detect NPU through OpenVINO: {e}")

        # Additional NPU detection for Intel systems
        if platform.system() == "Windows" and "intel" in self.cpu_info["brand"].lower():
            try:
                # Check for Intel NPU in device manager
                result = subprocess.run(
                    ["powershell", "-Command", "Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like '*NPU*' -or $_.Name -like '*Neural*'}"],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0 and result.stdout.strip():
                    npu_info["available"] = True
                    npu_info["intel_npu"] = True
                    if not npu_info["devices"]:  # Only add if not already detected
                        npu_info["devices"].append({
                            "id": 0,
                            "name": "Intel NPU",
                            "vendor": "Intel"
                        })

            except Exception as e:
                logger.warning(f"Could not detect Intel NPU: {e}")

        return npu_info

    def is_hardware_supported(self, hardware_type: str) -> bool:
        """Check if hardware type is supported"""
        if hardware_type.lower() == "cpu":
            return True  # CPU is always available
        elif hardware_type.lower() == "gpu":
            return self.gpu_info["available"]
        elif hardware_type.lower() == "npu":
            return self.npu_info["available"]
        else:
            return False

    def get_supported_formats(self, hardware_type: str) -> List[str]:
        """Get supported formats for hardware type"""
        if not self.is_hardware_supported(hardware_type):
            return []

        if hardware_type.lower() == "cpu":
            return ["hf", "onnx", "ov"]
        elif hardware_type.lower() == "gpu":
            if self.gpu_info["nvidia_gpu"]:
                return ["hf", "onnx", "ov"]
            elif self.gpu_info["intel_gpu"]:
                return ["hf", "ov"]  # Intel GPU might have limited ONNX support
            else:
                return ["hf"]
        elif hardware_type.lower() == "npu":
            return ["ov"]  # NPU typically only supports OpenVINO
        else:
            return []

    def get_hardware_memory(self, hardware_type: str) -> Optional[float]:
        """Get available memory for hardware type in GB"""
        if hardware_type.lower() == "cpu":
            return self.system_info["memory_gb"]
        elif hardware_type.lower() == "gpu":
            return self.gpu_info["total_memory_gb"] if self.gpu_info["available"] else None
        elif hardware_type.lower() == "npu":
            return 8.0 if self.npu_info["available"] else None  # Typical NPU memory
        else:
            return None

    def print_hardware_summary(self):
        """Print a summary of detected hardware"""
        print("\n=== Hardware Detection Summary ===")
        print(f"System: {self.system_info['platform']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")

        print(f"\nCPU: {self.cpu_info['brand']}")
        print(f"Cores: {self.cpu_info['count']}")
        print(f"Architecture: {self.cpu_info['arch']}")

        if self.gpu_info["available"]:
            print(f"\nGPU: Available")
            for device in self.gpu_info["devices"]:
                print(f"  - {device['name']} ({device['vendor']})")
                if device["memory_gb"] > 0:
                    print(f"    Memory: {device['memory_gb']} GB")
        else:
            print(f"\nGPU: Not available")

        if self.npu_info["available"]:
            print(f"\nNPU: Available")
            for device in self.npu_info["devices"]:
                print(f"  - {device['name']} ({device['vendor']})")
        else:
            print(f"\nNPU: Not available")

        print("\n=== Format Support by Hardware ===")
        for hw_type in ["cpu", "gpu", "npu"]:
            if self.is_hardware_supported(hw_type):
                formats = self.get_supported_formats(hw_type)
                print(f"{hw_type.upper()}: {', '.join(formats)}")
            else:
                print(f"{hw_type.upper()}: Not supported by hardware")


def get_hardware_detector() -> HardwareDetector:
    """Get hardware detector instance"""
    return HardwareDetector()


if __name__ == "__main__":
    # Test hardware detection
    detector = HardwareDetector()
    detector.print_hardware_summary()
