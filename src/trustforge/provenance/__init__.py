"""
TrustForge provenance infrastructure.

This package owns cryptographic hashing, Git execution provenance, execution
identity, native execution-manifest construction, and contract validation.

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
from trustforge.provenance.manifest import (
    ManifestConstructionError,
    ManifestWriteError,
    build_native_manifest,
    write_manifest_json,
)
from trustforge.provenance.validation import (
    ContractSchemaError,
    ContractValidationError,
    ValidationIssue,
    collect_contract_issues,
    load_contract_schema,
    validate_contract,
    validate_contract_file,
    validate_manifest_contract,
)

__all__ = [
    "ContractSchemaError",
    "ContractValidationError",
    "ExecutionIdentityError",
    "GitProvenanceError",
    "GitState",
    "ManifestConstructionError",
    "ManifestWriteError",
    "ProvenanceHashingError",
    "ValidationIssue",
    "build_execution_id",
    "build_native_manifest",
    "canonical_json_bytes",
    "capture_git_state",
    "collect_contract_issues",
    "format_execution_timestamp",
    "load_contract_schema",
    "sha256_bytes",
    "sha256_canonical_json",
    "sha256_file",
    "sha256_mapping",
    "sha256_text",
    "validate_contract",
    "validate_contract_file",
    "validate_manifest_contract",
    "write_manifest_json",
]
