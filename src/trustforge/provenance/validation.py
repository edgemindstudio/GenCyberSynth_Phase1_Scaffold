"""
TrustForge contract validation.

TrustForge contract schemas under ``schemas/`` are intentionally lightweight
YAML descriptions rather than JSON Schema documents.

This module validates runtime records against that contract format.

Supported schema field types
----------------------------

string
integer
number
boolean
enum
object
list

Supported contract features
---------------------------

required:
    Lists required field names.

fields:
    Defines child fields for an object.

allowed:
    Defines allowed values for an enum.

item_type:
    Defines the expected type of list items.

optional:
    Documentation-level marker. Requiredness is controlled by the parent's
    ``required`` list.

The validator is deliberately strict about declared required fields and type
compatibility, while allowing additional fields unless the contract later
introduces an explicit prohibition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ContractSchemaError(ValueError):
    """Raised when a TrustForge contract schema itself is malformed."""


class ContractValidationError(ValueError):
    """Raised when a record does not satisfy a TrustForge contract."""


@dataclass(frozen=True)
class ValidationIssue:
    """One contract-validation failure."""

    path: str
    message: str

    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"

        return self.message


def load_contract_schema(
    path: Path | str,
) -> dict[str, Any]:
    """
    Load a TrustForge YAML contract schema.

    The loaded schema must be a mapping containing:

    - schema_version
    - schema_type
    - required
    - fields
    """

    schema_path = Path(path).expanduser().resolve(strict=False)

    if not schema_path.is_file():
        raise ContractSchemaError(
            f"Contract schema does not exist: {schema_path}"
        )

    try:
        data = yaml.safe_load(
            schema_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ContractSchemaError(
            f"Unable to load contract schema: {schema_path}"
        ) from exc

    if not isinstance(data, Mapping):
        raise ContractSchemaError(
            "Contract schema root must be a mapping."
        )

    required_keys = {
        "schema_version",
        "schema_type",
        "required",
        "fields",
    }

    missing = sorted(
        required_keys.difference(data.keys())
    )

    if missing:
        raise ContractSchemaError(
            "Contract schema is missing required schema keys: "
            + ", ".join(missing)
        )

    if not isinstance(data["required"], list):
        raise ContractSchemaError(
            "Contract schema 'required' must be a list."
        )

    if not isinstance(data["fields"], Mapping):
        raise ContractSchemaError(
            "Contract schema 'fields' must be a mapping."
        )

    return dict(data)


def _is_integer(value: Any) -> bool:
    """
    Return True only for real integer values.

    bool is excluded because bool is a subclass of int in Python.
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """
    Return True for integer or floating-point numeric values, excluding bool.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _type_matches(
    value: Any,
    expected_type: str,
) -> bool:
    """Return whether a runtime value matches a TrustForge schema type."""

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "integer":
        return _is_integer(value)

    if expected_type == "number":
        return _is_number(value)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "object":
        return isinstance(value, Mapping)

    if expected_type == "list":
        return isinstance(value, list)

    if expected_type == "enum":
        return True

    raise ContractSchemaError(
        f"Unsupported TrustForge schema type: {expected_type!r}"
    )


def _validate_value(
    value: Any,
    field_schema: Mapping[str, Any],
    *,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    """Validate one value recursively against one field schema."""

    expected_type = field_schema.get("type")

    if not isinstance(expected_type, str):
        raise ContractSchemaError(
            f"Schema field {path!r} does not define a string type."
        )

    if not _type_matches(value, expected_type):
        issues.append(
            ValidationIssue(
                path,
                f"expected {expected_type}, "
                f"got {type(value).__name__}",
            )
        )
        return

    if expected_type == "enum":
        allowed = field_schema.get("allowed")

        if not isinstance(allowed, list):
            raise ContractSchemaError(
                f"Enum field {path!r} must define an allowed list."
            )

        if value not in allowed:
            issues.append(
                ValidationIssue(
                    path,
                    f"value {value!r} is not one of {allowed!r}",
                )
            )

        return

    if expected_type == "object":
        child_fields = field_schema.get(
            "fields",
            {},
        )

        child_required = field_schema.get(
            "required",
            [],
        )

        if not isinstance(child_fields, Mapping):
            raise ContractSchemaError(
                f"Object field {path!r} has invalid fields definition."
            )

        if not isinstance(child_required, list):
            raise ContractSchemaError(
                f"Object field {path!r} has invalid required definition."
            )

        for required_name in child_required:
            if required_name not in value:
                issues.append(
                    ValidationIssue(
                        f"{path}.{required_name}",
                        "required field is missing",
                    )
                )

        for child_name, child_schema in child_fields.items():
            if child_name not in value:
                continue

            if not isinstance(child_schema, Mapping):
                raise ContractSchemaError(
                    f"Schema definition for "
                    f"{path}.{child_name} must be a mapping."
                )

            _validate_value(
                value[child_name],
                child_schema,
                path=f"{path}.{child_name}",
                issues=issues,
            )

        return

    if expected_type == "list":
        item_type = field_schema.get("item_type")

        if item_type is None:
            return

        if not isinstance(item_type, str):
            raise ContractSchemaError(
                f"List field {path!r} has invalid item_type."
            )

        item_schema: dict[str, Any] = {
            "type": item_type,
        }

        for index, item in enumerate(value):
            _validate_value(
                item,
                item_schema,
                path=f"{path}[{index}]",
                issues=issues,
            )


def collect_contract_issues(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> list[ValidationIssue]:
    """
    Return all discovered validation issues without raising.

    Schema-definition errors still raise immediately because they indicate a
    broken TrustForge contract rather than invalid runtime data.
    """

    if not isinstance(record, Mapping):
        raise ContractValidationError(
            "Record must be a mapping."
        )

    if not isinstance(schema, Mapping):
        raise ContractSchemaError(
            "Schema must be a mapping."
        )

    required = schema.get("required")
    fields = schema.get("fields")

    if not isinstance(required, list):
        raise ContractSchemaError(
            "Schema 'required' must be a list."
        )

    if not isinstance(fields, Mapping):
        raise ContractSchemaError(
            "Schema 'fields' must be a mapping."
        )

    issues: list[ValidationIssue] = []

    for required_name in required:
        if required_name not in record:
            issues.append(
                ValidationIssue(
                    str(required_name),
                    "required field is missing",
                )
            )

    for field_name, field_schema in fields.items():
        if field_name not in record:
            continue

        if not isinstance(field_schema, Mapping):
            raise ContractSchemaError(
                f"Schema definition for {field_name!r} "
                "must be a mapping."
            )

        _validate_value(
            record[field_name],
            field_schema,
            path=str(field_name),
            issues=issues,
        )

    return issues


def validate_contract(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """
    Validate a record against a loaded TrustForge contract.

    Raises
    ------
    ContractValidationError
        If one or more contract violations are found.
    """

    issues = collect_contract_issues(
        record,
        schema,
    )

    if not issues:
        return

    detail = "\n".join(
        f"- {issue}"
        for issue in issues
    )

    raise ContractValidationError(
        "TrustForge contract validation failed:\n"
        + detail
    )


def validate_contract_file(
    record: Mapping[str, Any],
    schema_path: Path | str,
) -> None:
    """Load a TrustForge schema file and validate a record against it."""

    schema = load_contract_schema(schema_path)

    validate_contract(
        record,
        schema,
    )


def validate_manifest_contract(
    manifest: Mapping[str, Any],
    schema_path: Path | str,
) -> None:
    """
    Validate a runtime manifest against the TrustForge manifest contract.

    In addition to structural validation, the record must declare:

        schema_type: manifest

    and its schema_version must match the contract schema version.
    """

    schema = load_contract_schema(
        schema_path
    )

    if schema.get("schema_type") != "manifest":
        raise ContractSchemaError(
            "Provided schema is not a manifest contract."
        )

    if manifest.get("schema_type") != "manifest":
        raise ContractValidationError(
            "Manifest must declare schema_type='manifest'."
        )

    expected_version = schema.get(
        "schema_version"
    )

    actual_version = manifest.get(
        "schema_version"
    )

    if actual_version != expected_version:
        raise ContractValidationError(
            "Manifest schema_version does not match contract: "
            f"expected {expected_version!r}, "
            f"got {actual_version!r}."
        )

    validate_contract(
        manifest,
        schema,
    )
