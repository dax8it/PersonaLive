import os
from typing import Optional

# Must be set before MPS kernels are first initialized. It is harmless on CUDA/CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def resolve_device(device: Optional[str] = "auto") -> torch.device:
    """Resolve auto/cuda/mps/cpu to an available torch.device.

    PersonaLive was originally CUDA-first. This helper keeps CUDA behavior intact
    while making Apple Silicon MPS a first-class fallback before CPU.
    """
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device(requested)

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{device}'. Use auto, cuda, mps, or cpu.")


def get_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    dtype_name = (dtype_name or "fp32").lower()
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp32", "float32", "full"):
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use fp32, fp16, or bf16.")


def safe_empty_cache(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def safe_synchronize(device: Optional[torch.device] = None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def supports_xformers(device: torch.device, requested: bool = True) -> bool:
    # xformers memory-efficient attention is CUDA-only in this project.
    return bool(requested and device.type == "cuda" and torch.cuda.is_available())


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    # CPU generators are the most portable. Diffusers' randn_tensor can use a CPU
    # generator and move noise to MPS/CUDA, avoiding MPS Generator edge cases.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError(f"Expected boolean value, got '{value}'")
