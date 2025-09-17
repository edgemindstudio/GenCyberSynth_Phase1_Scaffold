# adapters/base.py

"""
Base interface for model adapters used by the GenCyberSynth scaffold.

An *adapter* is a thin wrapper around a specific generative model family
(e.g., GAN, VAE, Diffusion, Autoregressive). Each adapter exposes a single
entrypoint, `synth(config) -> manifest`, which generates (or locates) class-
conditional synthetic images and returns a manifest describing the outputs.

Concrete adapters (e.g., `GANAdapter`, `VAEAdapter`) should:
  1) Implement `synth(self, config) -> Manifest`.
  2) Write PNGs (or other image files) to
     `{artifacts}/{model_name}/synthetic/<class>/<seed>/...`.
  3) Return a validated manifest describing the files they produced.

The *manifest* must be compatible with downstream loaders and evaluators.
A minimal, valid manifest has the structure:

{
  "dataset": "USTC-TFC2016_40x40_gray",
  "seed": 42,
  "per_class_counts": {"0": 25, "1": 25, ...},
  "paths": [
    {"path": "artifacts/gan/synthetic/0/42/img_00001.png", "label": 0},
    ...
  ],
  # optional but recommended:
  "created_at": "2025-09-17T00:12:34"
}

This module defines:
  • TypedDicts describing the manifest schema.
  • An abstract `Adapter` base class with helpers to validate manifests.

Concrete adapters can import and reuse `Adapter.validate_manifest(...)`
to ensure their outputs are well-formed.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, TypedDict, Any, Mapping, Optional


# --------------------------------------------------------------------------- #
# Manifest schema (types)
# --------------------------------------------------------------------------- #

class ManifestPathItem(TypedDict):
    """One file entry in the manifest's `paths` list."""
    path: str   # relative or absolute path to an image file
    label: int  # integer class id for the image


class ManifestOptional(TypedDict, total=False):
    """Optional manifest keys supported by the scaffold."""
    created_at: str  # ISO 8601 timestamp
    notes: str       # freeform text


class Manifest(ManifestOptional):
    """Required manifest structure returned by adapters."""
    dataset: str
    seed: int
    per_class_counts: Dict[str, int]
    paths: List[ManifestPathItem]


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #

@dataclass
class AdapterInfo:
    """
    Lightweight metadata about an adapter implementation.

    `name` is used to route CLI calls (e.g., `--model gan`).
    """
    name: str


class Adapter(ABC):
    """
    Abstract base class for all model adapters.

    Concrete adapters MUST implement `synth(config) -> Manifest`.

    Recommended usage in subclasses:
        class GANAdapter(Adapter):
            info = AdapterInfo(name="gan")

            def synth(self, config: dict) -> Manifest:
                # ... generate or load images ...
                manifest: Manifest = {...}
                self.validate_manifest(manifest)
                return manifest
    """

    # Adapter routing info; override in subclasses (e.g., AdapterInfo("gan")).
    info: AdapterInfo = AdapterInfo(name="base")

    # ---- Required API ---------------------------------------------------- #
    @abstractmethod
    def synth(self, config: Mapping[str, Any]) -> Manifest:
        """
        Generate class-conditional images and return a manifest.

        Implementations should:
          • respect relevant config keys (datasets, artifacts root, seeds, etc.)
          • write images to disk
          • build the manifest (dataset, seed, per_class_counts, paths)
          • call `self.validate_manifest(manifest)` before returning

        Args:
            config: Configuration mapping (parsed from YAML or similar).

        Returns:
            A validated `Manifest` describing the generated files.
        """
        raise NotImplementedError

    # ---- Helpers shared by all adapters --------------------------------- #
    def validate_manifest(self, manifest: Manifest) -> None:
        """
        Validate the shape and basic invariants of a manifest.

        This function raises `ValueError` with a helpful message if something
        is malformed. Adapters should call this before returning their manifest.

        Checks performed:
          • Required keys exist with correct types.
          • `paths` entries each have `path: str` and `label: int`.
          • `per_class_counts` has non-negative ints keyed by strings.
          • The distribution of labels in `paths` matches `per_class_counts`.
        """
        # Required keys
        required_keys = ("dataset", "seed", "per_class_counts", "paths")
        for k in required_keys:
            if k not in manifest:
                raise ValueError(f"manifest missing required key: '{k}'")

        # Types: dataset, seed
        if not isinstance(manifest["dataset"], str) or not manifest["dataset"]:
            raise ValueError("manifest['dataset'] must be a non-empty string")
        if not isinstance(manifest["seed"], int):
            raise ValueError("manifest['seed'] must be an int")

        # Types: per_class_counts
        pcc = manifest["per_class_counts"]
        if not isinstance(pcc, dict) or not all(isinstance(k, str) for k in pcc.keys()):
            raise ValueError("manifest['per_class_counts'] must be Dict[str, int]")
        if not all(isinstance(v, int) and v >= 0 for v in pcc.values()):
            raise ValueError("manifest['per_class_counts'] values must be non-negative ints")

        # Types: paths
        paths = manifest["paths"]
        if not isinstance(paths, list):
            raise ValueError("manifest['paths'] must be a list")
        for i, item in enumerate(paths):
            if not isinstance(item, dict):
                raise ValueError(f"manifest['paths'][{i}] must be a dict")
            if "path" not in item or "label" not in item:
                raise ValueError(f"manifest['paths'][{i}] must contain 'path' and 'label'")
            if not isinstance(item["path"], str) or not item["path"]:
                raise ValueError(f"manifest['paths'][{i}]['path'] must be a non-empty string")
            if not isinstance(item["label"], int):
                raise ValueError(f"manifest['paths'][{i}]['label'] must be an int")

        # Invariant: label counts match per_class_counts
        calculated: Dict[str, int] = {}
        for item in paths:
            key = str(item["label"])
            calculated[key] = calculated.get(key, 0) + 1

        # Compare only for keys that appear in either mapping
        all_keys = set(pcc.keys()) | set(calculated.keys())
        mismatches: List[str] = []
        for k in sorted(all_keys, key=lambda x: (len(x), x)):
            left = pcc.get(k, 0)
            right = calculated.get(k, 0)
            if left != right:
                mismatches.append(f"class '{k}': per_class_counts={left}, paths={right}")

        if mismatches:
            raise ValueError(
                "manifest label distribution mismatch:\n  " + "\n  ".join(mismatches)
            )

        # Optional keys (when present) should be well-typed.
        if "created_at" in manifest and not isinstance(manifest["created_at"], str):
            raise ValueError("manifest['created_at'] must be a string (ISO timestamp)")
        if "notes" in manifest and not isinstance(manifest["notes"], str):
            raise ValueError("manifest['notes'] must be a string")


__all__ = [
    "Adapter",
    "AdapterInfo",
    "Manifest",
    "ManifestPathItem",
]
