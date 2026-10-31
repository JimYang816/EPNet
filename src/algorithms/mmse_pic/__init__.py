"""Fixed-CSI, uncoded soft MMSE parallel interference cancellation."""
from .detector import equalize_batch, mmse_pic

__all__ = ["equalize_batch", "mmse_pic"]
