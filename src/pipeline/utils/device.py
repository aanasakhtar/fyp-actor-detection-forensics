"""
Hardware runtime and device detection utilities.
"""

import os
from typing import Dict, Any, List

def select_device(preferred: str = "auto") -> Any:
    """Select appropriate PyTorch device based on hardware availability."""
    try:
        import torch
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return "cpu"

def get_onnx_providers() -> List[str]:
    """Return available ONNX Runtime execution providers in priority order."""
    providers = []
    try:
        import torch
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")
    return providers

def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed information about current compute environment and GPUs."""
    info = {
        "cuda_available": False,
        "device_count": 0,
        "current_device": None,
        "device_name": "CPU",
        "total_memory_gb": None,
        "allocated_memory_gb": None,
        "cached_memory_gb": None
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = idx
            info["device_name"] = props.name
            info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["allocated_memory_gb"] = round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 2)
            info["cached_memory_gb"] = round(torch.cuda.memory_reserved(idx) / (1024 ** 3), 2)
    except Exception:
        pass
        
    return info
