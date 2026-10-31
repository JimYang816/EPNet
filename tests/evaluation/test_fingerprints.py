"""Source identities remain compatible with Python 3.10 model checkpoints."""
from __future__ import annotations

import ast
import sys

import pytest

from src.evaluation.fingerprints import normalized_source


def test_python310_function_identity_on_newer_python():
    expected = ("Module(body=[FunctionDef(name='x', args=arguments(posonlyargs=[], args=[], "
                "kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Pass()], decorator_list=[])], type_ignores=[])")
    assert normalized_source("def x():\n pass") == expected
    assert normalized_source('def x():\n "ignored documentation"\n pass') == expected


def test_python310_class_identity_on_newer_python():
    expected = "Module(body=[ClassDef(name='X', bases=[], keywords=[], body=[Pass()], decorator_list=[])], type_ignores=[])"
    assert normalized_source("class X:\n pass") == expected


def test_computation_changes_still_invalidate_source():
    assert normalized_source("x = 1") != normalized_source("x = 2")
    assert normalized_source("def x(a):\n return a") != normalized_source("def x(a):\n return -a")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Generic functions require Python 3.12")
def test_nonempty_type_parameters_are_preserved():
    source = "def identity[T](x: T) -> T:\n return x"
    assert "type_params=" in normalized_source(source)
    assert "TypeVar" in ast.dump(ast.parse(source))
