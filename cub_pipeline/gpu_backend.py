import os
import site
from typing import Any


def gpu_enabled() -> bool:
    return os.environ.get("BIRD_PIPELINE_USE_GPU", "1").strip().lower() not in {"0", "false", "off", "no"}


def configure_cuda_library_path() -> bool:
    """
    Prepend CUDA library directories from pip-installed NVIDIA wheels to
    LD_LIBRARY_PATH so FAISS GPU can load symbols reliably.
    """
    if not gpu_enabled():
        return False
    roots: list[str] = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.append(user_site)
    except Exception:
        pass
    try:
        for p in site.getsitepackages():
            if isinstance(p, str):
                roots.append(p)
    except Exception:
        pass

    candidates: list[str] = []
    for root in roots:
        nvidia_root = os.path.join(root, "nvidia")
        for rel in ("cublas/lib", "cuda_runtime/lib", "cuda_nvrtc/lib"):
            lib_path = os.path.join(nvidia_root, rel)
            if os.path.isdir(lib_path):
                candidates.append(lib_path)
    if not candidates:
        return False

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged: list[str] = []
    for p in candidates + existing:
        if p not in merged:
            merged.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    return True


def load_cupy() -> Any | None:
    if not gpu_enabled():
        return None
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
    except Exception:
        return None
    return cp


def load_cuml() -> Any | None:
    if not gpu_enabled():
        return None
    try:
        import cuml  # type: ignore
    except Exception:
        return None
    return cuml


def faiss_gpu_available(faiss_module: Any) -> bool:
    if not gpu_enabled():
        return False
    try:
        has_gpu = hasattr(faiss_module, "StandardGpuResources")
        if not has_gpu:
            return False
        if hasattr(faiss_module, "get_num_gpus"):
            return int(faiss_module.get_num_gpus()) > 0
        return True
    except Exception:
        return False

