"""LoRango SDXL pair-training toolkit."""

import os


# Must be set before the first CUDA context is initialized. Strict deterministic
# training also disables the non-deterministic SDP kernels in training.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.4.1"
