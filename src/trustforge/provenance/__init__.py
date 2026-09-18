"""
TrustForge provenance primitives.

This package owns cryptographic hashing, Git execution provenance, execution
identity, and eventually manifest construction.

Historical provenance is preserved exactly as originally recorded. New
TrustForge-native provenance uses SHA256 and explicit Git working-tree state.
"""

from trustforge.provenance.execution import (
    ExecutionIdentityError,
    build_execution_id,
    format_execution_timestamp,
)
from trustforge.provenance.git import (
    GitProvenanceError,
    GitState,
    capture_git_state,
)
from trustforge.provenance.hashing import (
    ProvenanceHashingError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    sha256_mapping,
    sha256_text,
)

__all__ = [
    "ExecutionIdentityError",
    "GitProvenanceError",
    "GitState",
    "ProvenanceHashingError",
    "build_execution_id",
    "canonical_json_bytes",
    "capture_git_state",
    "format_execution_timestamp",
    "sha256_bytes",
    "sha256_canonical_json",
    "sha256_file",
    "sha256_mapping",
    "sha256_text",
]
