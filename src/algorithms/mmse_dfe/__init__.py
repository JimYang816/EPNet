"""Full-matrix MMSE decision-feedback equalization with fixed CSI."""
from .detector import equalize_batch, mmse_dfe

__all__ = ["equalize_batch", "mmse_dfe"]
