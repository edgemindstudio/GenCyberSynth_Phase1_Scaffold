"""Tests for TrustForge provenance hashing."""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from trustforge.provenance import (
    ProvenanceHashingError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    sha256_mapping,
    sha256_text,
)


class ProvenanceHashingTests(unittest.TestCase):
    """Validate deterministic TrustForge SHA256 behavior."""

    def test_sha256_bytes_matches_hashlib(self) -> None:
        data = b"trustforge"

        self.assertEqual(
            sha256_bytes(data),
            hashlib.sha256(data).hexdigest(),
        )

    def test_sha256_text_matches_encoded_bytes(self) -> None:
        text = "TrustForge provenance"

        self.assertEqual(
            sha256_text(text),
            sha256_bytes(text.encode("utf-8")),
        )

    def test_sha256_file_hashes_exact_file_bytes(self) -> None:
        content = b"exact artifact bytes\n"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(content)

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(content).hexdigest(),
            )

    def test_canonical_json_sorts_mapping_keys(self) -> None:
        left = {
            "model": "gan",
            "seed": 42,
        }

        right = {
            "seed": 42,
            "model": "gan",
        }

        self.assertEqual(
            canonical_json_bytes(left),
            canonical_json_bytes(right),
        )

    def test_mapping_hash_is_order_independent(self) -> None:
        left = {
            "model": {
                "name": "gan",
                "latent_dim": 100,
            },
            "seed": 42,
        }

        right = {
            "seed": 42,
            "model": {
                "latent_dim": 100,
                "name": "gan",
            },
        }

        self.assertEqual(
            sha256_mapping(left),
            sha256_mapping(right),
        )

    def test_canonical_json_rejects_nan(self) -> None:
        with self.assertRaises(ProvenanceHashingError):
            sha256_canonical_json(
                {"value": math.nan}
            )

    def test_canonical_json_rejects_unserializable_value(self) -> None:
        with self.assertRaises(ProvenanceHashingError):
            sha256_canonical_json(
                {"path": Path("/tmp/example")}
            )


if __name__ == "__main__":
    unittest.main()
