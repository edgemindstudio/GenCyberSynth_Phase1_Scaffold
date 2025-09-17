# adapters/registry.py

"""
Adapter registry
----------------
Central place to register and retrieve concrete data-generation adapters.

Usage
-----
from adapters.registry import make_adapter, list_adapters

adapter = make_adapter("diffusion")
manifest = adapter.synth(cfg)

Design
------
- Lightweight, import-safe: each adapter import is wrapped in try/except so the
  registry can be imported even if some backends are not installed yet.
- Extensible: call `register_adapter("name", AdapterSubclass)` at import-time
  (e.g., inside your adapter module) or add it here.
- Friendly errors: unknown names produce a clear message that lists available
  adapters and hints at missing dependencies.

Add new adapters
----------------
1) Implement `class MyAdapter(Adapter): ...` in `adapters/my_adapter.py`.
2) Register it here (preferred) or inside your module:
       from adapters.registry import register_adapter
       register_adapter("my_adapter", MyAdapter)
"""

from __future__ import annotations

from typing import Dict, Type, List, Optional

from .base import Adapter

# Internal mutable registry (name -> Adapter subclass)
_REGISTRY: Dict[str, Type[Adapter]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_adapter(name: str, cls: Type[Adapter]) -> None:
    """
    Register an adapter under a given name.

    Overwrites any existing entry with the same name (useful for hot-reload / dev).
    """
    if not issubclass(cls, Adapter):
        raise TypeError(f"Adapter class for '{name}' must subclass adapters.base.Adapter")
    _REGISTRY[name] = cls


def make_adapter(name: str, **kwargs) -> Adapter:
    """
    Instantiate a registered adapter by name.

    Raises a KeyError with a helpful message if the adapter is unknown.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(
            f"Unknown adapter '{name}'. Available adapters: {available}. "
            "If this adapter should exist, ensure its module imports without errors "
            "and that it calls register_adapter(...)."
        )
    return cls(**kwargs)  # type: ignore[call-arg]


def list_adapters() -> List[str]:
    """Return a sorted list of registered adapter names."""
    return sorted(_REGISTRY.keys())


def get_registry() -> Dict[str, Type[Adapter]]:
    """Return a (shallow) copy of the registry mapping."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Best-effort imports of built-in adapters
# (Each is optional; failures do not break the registry import.)
# ---------------------------------------------------------------------------
def _try_register(name: str, import_path: str, class_name: str) -> Optional[str]:
    try:
        module = __import__(import_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        register_adapter(name, cls)
        return None
    except Exception as e:
        # Return a brief reason; we do not print by default to avoid noisy imports.
        # You can log this in your app if you want visibility into skipped adapters.
        return f"{name} → skipped ({e.__class__.__name__}: {e})"


# Built-ins: add more as you implement them
_skip_notes: List[str] = []
_skip_notes.append(_try_register("diffusion", "adapters.diffusion_adapter", "DiffusionAdapter"))
# Examples for future adapters (uncomment when available):
# _skip_notes.append(_try_register("gan",         "adapters.gan_adapter",         "GANAdapter"))
# _skip_notes.append(_try_register("vae",         "adapters.vae_adapter",         "VAEAdapter"))
# _skip_notes.append(_try_register("autoregressive","adapters.autoregressive_adapter","AutoregressiveAdapter"))
# _skip_notes.append(_try_register("maf",         "adapters.maf_adapter",         "MAFAdapter"))
# _skip_notes.append(_try_register("gmm",         "adapters.gmm_adapter",         "GMMAdapter"))
# _skip_notes.append(_try_register("rbm",         "adapters.rbm_adapter",         "RBMAdapter"))

# Clean out Nones from the notes list in case the caller wants to inspect it
SKIPPED_IMPORTS: List[str] = [n for n in _skip_notes if n]

__all__ = [
    "register_adapter",
    "make_adapter",
    "list_adapters",
    "get_registry",
    "SKIPPED_IMPORTS",
]

