"""Device selection — GPU when usable, CPU otherwise."""
import os
import warnings

import torch


def _cuda_is_usable() -> bool:
    """True only if CUDA is present AND can actually run a kernel.

    torch.cuda.is_available() can return True for a GPU whose compute
    capability is newer than the installed torch build supports (e.g. an
    RTX 50-series / sm_120 card with a cu121 wheel). Such a GPU is
    "detected" but every kernel launch raises. We probe with a tiny op so
    callers transparently fall back to CPU instead of crashing mid-run.
    Set DEEPHATE_FORCE_CPU=1 to skip CUDA entirely.
    """
    if os.environ.get("DEEPHATE_FORCE_CPU") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.zeros(1, device="cuda:0") + 1).cpu()
        return True
    except Exception as exc:  # noqa: BLE001 - any CUDA failure -> use CPU
        warnings.warn(
            f"CUDA detected but not usable ({exc}); falling back to CPU. "
            "Install a torch build matching your GPU to enable GPU acceleration.",
            RuntimeWarning,
        )
        return False


def get_device() -> torch.device:
    if _cuda_is_usable():
        return torch.device("cuda:0")
    return torch.device("cpu")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
