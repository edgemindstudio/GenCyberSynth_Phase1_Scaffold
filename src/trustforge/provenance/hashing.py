"""
Cryptographic hashing primitives for TrustForge provenance.

TrustForge-native provenance uses SHA256.

Historical studies may contain other digest algorithms, such as the SHA1
configuration digest recorded by the original Paper 1 execution system.
Those historical values must be preserved exactly as recorded rather than
silently converted into TrustForge-native hashes.

This module distinguishes two important concepts:

1. exact file hashing:
       hashes the bytes physically stored in a file;

2. canonical structured-data hashing:
       serializes a JSON-compatible object deterministically and hashes that
       representation.

Those hashes answer different provenance questions and must not be treated as
interchangeable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CHUNK_SIZE = 1024 * 1024


class ProvenanceHashingError(ValueError):
    """Raised when provenance input cannot be hashed deterministically."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hexadecimal digest of exact bytes."""

    if not isinstance(data, bytes):
        raise ProvenanceHashingError(
            "sha256_bytes requires a bytes value."
        )

    return hashlib.sha256(data).hexdigest()


def sha256_text(
    text: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Return the SHA256 hexadecimal digest of encoded text."""

    if not isinstance(text, str):
        raise ProvenanceHashingError(
            "sha256_text requires a string value."
        )

    return sha256_bytes(text.encode(encoding))


def sha256_file(
    path: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Return the SHA256 digest of the exact bytes stored in a file.

    The file is streamed in chunks so large artifacts do not need to be loaded
    completely into memory.
    """

    if chunk_size <= 0:
        raise ProvenanceHashingError(
            "chunk_size must be greater than zero."
        )

    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise ProvenanceHashingError(
            f"Cannot hash missing or non-file path: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """
    Serialize JSON-compatible data into deterministic UTF-8 bytes.

    Dictionary keys are sorted and unnecessary whitespace is removed.
    NaN and Infinity are rejected because they are not portable canonical JSON
    values.

    This function does not mutate the supplied object.
    """

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceHashingError(
            "Value cannot be represented as canonical JSON."
        ) from exc

    return text.encode("utf-8")


def sha256_canonical_json(value: Any) -> str:
    """Return the SHA256 digest of a canonical JSON representation."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_mapping(mapping: Mapping[str, Any]) -> str:
    """
    Return a deterministic SHA256 digest for a configuration-like mapping.

    This is a semantic structured-data hash. It is not the same thing as
    hashing the original YAML, JSON, or TOML file bytes.
    """

    if not isinstance(mapping, Mapping):
        raise ProvenanceHashingError(
            "sha256_mapping requires a mapping."
        )

    return sha256_canonical_json(dict(mapping))
