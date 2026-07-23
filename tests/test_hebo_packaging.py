"""Regression tests for HEBO's separately managed dependency status."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


def _canonical_requirement_name(requirement: str) -> str:
    """Extract and normalize the distribution name from a PEP 508 requirement."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, f"Could not parse dependency requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def test_optional_dependencies_do_not_declare_hebo():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        optional_dependencies = tomllib.load(pyproject_file)["project"][
            "optional-dependencies"
        ]

    declarations = [
        (extra, requirement)
        for extra, requirements in optional_dependencies.items()
        for requirement in requirements
        if _canonical_requirement_name(requirement) == "hebo"
    ]

    assert declarations == [], (
        "HEBO conflicts with dfbench's dependency stack and must remain an "
        f"unbundled backend; found declarations: {declarations}"
    )


def test_hebo_constructor_explains_unbundled_backend(monkeypatch):
    from dfbench.algorithms.surrogate_based import hebo_bo

    backend_error = ImportError("incompatible HEBO backend")
    monkeypatch.setattr(hebo_bo, "_HEBO_AVAILABLE", False)
    monkeypatch.setattr(hebo_bo, "_HEBO_IMPORT_ERROR", backend_error)

    with pytest.raises(ImportError) as error:
        hebo_bo.HEBO()

    message = str(error.value).lower()
    assert "dfbench[" not in message
    assert "compatible" in message
    assert "not bundled" in message or "separately" in message
    assert error.value.__cause__ is backend_error
