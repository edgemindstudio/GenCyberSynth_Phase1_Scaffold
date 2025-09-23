# diffusion/__init__.py

# Expose a stable synth entrypoint to adapters
from .sample import synth  # noqa: F401
__all__ = ["synth"]