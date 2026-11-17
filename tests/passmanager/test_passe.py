"""Tests for the pass declaration and instance helpers."""

import ast
import doctest
import sys

import pytest
from attr.exceptions import FrozenInstanceError

from libstatic.passmanager._model import Forest, Level, PassKind, Tree
from libstatic.passmanager._passe import (
    PassInstance,
    PassPrototype,
    analysis,
    new_pass_prototype,
    transformation,
)


def _simple_pass(connector, element):
    return {}


def test_pass_instance_reports_missing_required_parameter():
    """Verify that an unconfigured pass reports its first required parameter."""

    def checker(connector, node, *, required, also_required):
        return {}

    proto = new_pass_prototype(checker, kind=PassKind.ANALYSIS, on=ast.Module)

    assert proto._instanciate().missing_param() == 'required'


def test_pass_instance_binds_required_positional_parameter():
    """Verify that a required pass parameter can be bound positionally."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )('value')

    assert instance.args == {'required': 'value'}


def test_pass_instance_binds_optional_keyword_parameters():
    """Verify that declared optional pass parameters can be overridden."""

    def checker(connector, node, *, option='default'):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )(option='custom')

    assert instance.args == {'option': 'custom'}


def test_pass_instance_rejects_too_many_positional_arguments():
    """Verify that pass instances reject more positionals than required parameters."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )._instanciate()

    with pytest.raises(TypeError, match='too many positional'):
        instance('one', 'two')


def test_pass_instance_rejects_too_many_keyword_arguments():
    """Verify that pass instances reject more keywords than declared parameters."""

    def checker(connector, node, *, required, optional='default'):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )._instanciate()

    with pytest.raises(TypeError, match='too many keyword parmeters'):
        instance(required='one', optional='two', extra='three')


def test_pass_instance_rejects_unknown_keyword_arguments():
    """Verify that pass instances reject undeclared parameter names."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )._instanciate()

    with pytest.raises(TypeError, match='unexpected argument'):
        instance(unknown=True)


def test_pass_instance_rejects_duplicate_parameter_values():
    """Verify that a required parameter cannot be supplied both positionally and by name."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )._instanciate()

    with pytest.raises(TypeError, match='got several values'):
        instance('first', required='second')


def test_pass_instance_reuses_instance_when_arguments_are_unchanged():
    """Verify that binding no new values or the same values preserves the instance."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )(required='value')

    assert instance() is instance
    assert instance(required='value') is instance


def test_pass_instance_can_replace_bound_arguments():
    """Verify that replacing a pass instance's argument mapping returns the updated mapping."""

    def checker(connector, node, *, required):
        return {}

    instance = new_pass_prototype(
        checker, kind=PassKind.ANALYSIS, on=ast.Module
    )(required='value')

    assert instance._replace(args={'required': 'replacement'}).args == {
        'required': 'replacement'
    }


def test_prototype_targets_ast_nodes():
    """Verify that AST pass targets are assigned the node execution level."""
    proto = new_pass_prototype(
        _simple_pass, kind=PassKind.TRANSFORMATION, on=ast.Module
    )

    assert proto.runs_on is Level.NODE
    assert proto.runs_on_type is ast.Module


def test_prototype_targets_forests():
    """Verify that forest pass targets are assigned the forest execution level."""
    proto = new_pass_prototype(_simple_pass, kind=PassKind.ANALYSIS, on=Forest)

    assert proto.runs_on is Level.FOREST
    assert proto.runs_on_type is Forest


def test_prototype_rejects_mixed_forest_and_node_targets():
    """Verify that a pass cannot target both forests and individual node types."""
    with pytest.raises(TypeError, match='cannot run both on'):
        new_pass_prototype(
            _simple_pass,
            kind=PassKind.TRANSFORMATION,
            on=(Forest, ast.Module),
        )


def test_prototype_rejects_empty_targets():
    """Verify that pass creation rejects an empty target collection."""
    with pytest.raises(ValueError, match='parameter "on" cannot be empty'):
        new_pass_prototype(
            _simple_pass, kind=PassKind.TRANSFORMATION, on=()
        )


def test_prototype_rejects_more_than_two_positional_function_arguments():
    """Verify that pass functions cannot declare a third positional argument."""

    def checker(connector, node, extra):
        return {}

    with pytest.raises(TypeError, match='more than two positional'):
        new_pass_prototype(checker, kind=PassKind.ANALYSIS, on=ast.Module)


def test_prototype_rejects_variable_length_function_arguments():
    """Verify that pass functions cannot declare variable-length arguments."""

    def checker(connector, node, *values):
        return {}

    with pytest.raises(TypeError, match='variable length arguments'):
        new_pass_prototype(checker, kind=PassKind.ANALYSIS, on=ast.Module)


def test_prototype_requires_two_positional_function_arguments():
    """Verify that pass functions must declare exactly two positional arguments."""

    def checker(connector):
        return {}

    with pytest.raises(TypeError, match='exactly'):
        new_pass_prototype(checker, kind=PassKind.ANALYSIS, on=ast.Module)


def test_prototype_wraps_uninspectable_function_errors():
    """Verify that pass creation reports callables whose signatures cannot be inspected."""
    with pytest.raises(TypeError, match="can't be inspected") as exc_info:
        new_pass_prototype(object(), kind=PassKind.ANALYSIS, on=ast.Module)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_prototype_normalizes_iterable_targets():
    """Verify that iterable target collections are converted and singletons flattened."""
    single_target = new_pass_prototype(
        _simple_pass, kind=PassKind.ANALYSIS, on=[ast.Module]
    )
    multiple_targets = new_pass_prototype(
        _simple_pass, kind=PassKind.ANALYSIS, on=[ast.Module, ast.expr]
    )

    assert single_target.runs_on_type is ast.Module
    assert multiple_targets.runs_on_type == (ast.Module, ast.expr)


def test_prototype_resolves_direct_dependencies():
    """Verify that named dependencies resolve to their pass prototypes."""
    dependency = new_pass_prototype(
        _simple_pass, kind=PassKind.ANALYSIS, on=ast.Module
    )
    parent = new_pass_prototype(
        _simple_pass,
        kind=PassKind.ANALYSIS,
        on=ast.Module,
        dependencies=(dependency.name,),
    )

    assert parent.get_dependencies({dependency.name: dependency}.__getitem__) == {
        dependency
    }


def test_prototype_collects_transitive_dependencies():
    """Verify that all nested pass dependencies are collected."""
    dependency = new_pass_prototype(
        _simple_pass, kind=PassKind.ANALYSIS, on=ast.Module
    )
    child = new_pass_prototype(
        _simple_pass,
        kind=PassKind.ANALYSIS,
        on=ast.Module,
        dependencies=(dependency,),
    )
    parent = new_pass_prototype(
        _simple_pass,
        kind=PassKind.TRANSFORMATION,
        on=ast.Module,
        dependencies=(child,),
    )

    assert parent.get_all_dependencies(lambda name: {child.name: child, dependency.name: dependency}[name]) == {
        child,
        dependency,
    }


def test_prototype_exposes_wrapped_function_metadata():
    """Verify that prototype introspection exposes its function metadata."""

    def documented_pass(connector, node):
        """Return pass data."""
        return {}

    proto = new_pass_prototype(
        documented_pass, kind=PassKind.TRANSFORMATION, on=ast.Module
    )

    assert proto.__doc__ == documented_pass.__doc__
    with pytest.raises(FrozenInstanceError):
        proto.__doc__ = 'new docstring'
    assert proto.__name__ == 'documented_pass'
    assert proto.__wrapped__ is documented_pass
    assert proto.proto is proto
    assert str(proto).endswith("'documented_pass'")


def test_prototype_can_replace_its_name():
    """Verify that replacing a prototype field returns a prototype with that value."""
    proto = new_pass_prototype(
        _simple_pass, kind=PassKind.TRANSFORMATION, on=ast.Module
    )

    assert proto._replace(name='renamed').name == 'renamed'


def test_prototype_reports_missing_required_parameter():
    """Verify that a prototype reports a required parameter before instantiation."""

    def checker(connector, node, *, required):
        return {}

    proto = new_pass_prototype(checker, kind=PassKind.ANALYSIS, on=ast.Module)

    assert proto.missing_param() == 'required'


def test_transformation_decorator_creates_node_pass():
    """Verify that the transformation decorator creates a node-level transformation."""

    @transformation(on=ast.Module)
    def transform(connector, node):
        return {}

    assert isinstance(transform, PassPrototype)
    assert transform.kind is PassKind.TRANSFORMATION
    assert transform.runs_on is Level.NODE
    assert transform.runs_on_type is ast.Module


def test_analysis_decorator_creates_tree_pass():
    """Verify that the analysis decorator creates a tree-level analysis."""

    @analysis(on=Tree)
    def analyse(connector, tree):
        return {}

    assert isinstance(analyse, PassPrototype)
    assert analyse.kind is PassKind.ANALYSIS
    assert analyse.runs_on is Level.TREE
    assert analyse.runs_on_type is Tree


def test_analysis_decorator_creates_forest_pass():
    """Verify that the analysis decorator creates a forest-level analysis."""

    @analysis(on=Forest)
    def forest_analysis(connector, forest):
        return {}

    assert isinstance(forest_analysis, PassPrototype)
    assert forest_analysis.kind is PassKind.ANALYSIS
    assert forest_analysis.runs_on is Level.FOREST
    assert forest_analysis.runs_on_type is Forest


def test_prototype_call_creates_configured_pass_instance():
    """Verify that calling a prototype binds the supplied pass-specific arguments."""

    def configured_pass(connector, node, *, required):
        return {'required': required}

    proto = new_pass_prototype(
        configured_pass, kind=PassKind.ANALYSIS, on=ast.Module
    )
    instance = proto(required='value')

    assert isinstance(instance, PassInstance)
    assert instance.args == {'required': 'value'}
    assert instance.missing_param() is None


def test_decorated_pass_docstrings_are_registered_for_doctest():
    """Verify that both pass decorators expose their function docstrings to doctest."""
    module = sys.modules[__name__]
    tests = getattr(module, '__test__', None)
    if tests is None:
        tests = {}
        module.__test__ = tests
    tests['existing_documentation'] = 'Existing module doctest entry.'

    @analysis(on=ast.Module)
    def documented_pass(connector, node):
        """A documented analysis.

        >>> 1 + 1
        2
        """
        return {'result': None}

    analysis_pass = documented_pass

    @transformation(on=ast.Module)
    def documented_pass(connector, node):
        """A documented transformation.

        >>> 2 + 2
        4
        """
        return {'update': False}

    transformation_pass = documented_pass

    assert tests['existing_documentation'] == 'Existing module doctest entry.'
    assert any(value is analysis_pass.__wrapped__ for value in tests.values()), tests
    assert any(value is transformation_pass.__wrapped__ for value in tests.values()), tests

    discovered = [
        test for test in doctest.DocTestFinder().find(module)
        if test.name.endswith('.documented_pass')
        or '.documented_pass#' in test.name
    ]
    assert len(discovered) == 2
    assert all(doctest.DocTestRunner().run(test).failed == 0 for test in discovered)