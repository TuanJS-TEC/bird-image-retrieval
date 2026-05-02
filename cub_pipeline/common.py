import os


def _debug_log(*args, **kwargs) -> None:
    """Debug instrumentation da duoc vo hieu hoa sau khi fix xong."""
    return None


def parallel_tier_extraction_workers() -> int:
    """So process worker cho Tang 1/2/3 (encode anh doc lap). Bien: CUB_TIER_EXTRACTION_WORKERS."""
    raw = int(os.environ.get("CUB_TIER_EXTRACTION_WORKERS", "0") or "0")
    if raw >= 1:
        return max(1, raw)
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu))
