"""Tests for pass execution, orchestration, and hooks."""

import ast

import pytest

from libstatic._lib.structures import FrozenDict
from libstatic.passmanager._model import Forest, Level, PassKind, Tree
from libstatic.passmanager._passe import analysis, transformation
from libstatic.passmanager._passmanager import (
    AnalysisRunner,
    PassCycleError,
    TransformationRunner,
    FailedPass,
    Hooks,
    PassContext,
    PassManager,
    PreparedPass,
    Runner,
    Trigger,
    UnpreparedPass,
    _add_tree,
    _remove_tree,
)


def _tree(identifier='mod', source='x = 1'):
    return Tree(ast.parse(source), identifier)


def _manager_with_tree():
    pm = PassManager()
    tree = _tree()
    pm.add(tree)
    return pm, tree


def test_pass_context_rejects_current_pass_when_empty():
    """Verify that a context without an active pass has no current pass run."""
    with pytest.raises(RuntimeError, match='no pass is currently running'):
        _ = PassContext()._current_passrun


def test_pass_context_detects_a_repeated_pass_run():
    """Verify that pushing the same pass run twice is reported as a cycle."""

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    forest = Forest()
    passrun = (inspect_node(), (forest,), {})
    context = PassContext()

    with context._push_pass(passrun):
        with pytest.raises(PassCycleError, match='cycle detected') as exc_info:
            with context._push_pass(passrun):
                pass

    assert exc_info.value.passrun[0] is passrun[0]
    assert exc_info.value.passrun[1] == passrun[1]
    assert exc_info.value.passrun[2] == passrun[2]
    assert isinstance(exc_info.value, RuntimeError)
    assert str(exc_info.value).startswith('cycle detected with pass:')


def test_runner_repeated_pass_run_is_catchable_as_cycle_error():
    """Verify that a recursive pass run raises the public cycle exception."""

    @analysis(on=Forest)
    def recursive_pass(connector, forest):
        return {'result': connector.gather(recursive_pass)}

    with pytest.raises(PassCycleError, match='cycle detected'):
        PassManager().run(recursive_pass)


def test_runner_detects_cycle_across_three_passes():
    """Verify that a cycle spanning three distinct passes is detected."""

    @analysis(on=Forest)
    def first_pass(connector, forest):
        return {'result': connector.gather(second_pass)}

    @analysis(on=Forest)
    def second_pass(connector, forest):
        return {'result': connector.gather(third_pass)}

    @analysis(on=Forest)
    def third_pass(connector, forest):
        return {'result': connector.gather(first_pass)}

    with pytest.raises(PassCycleError, match='cycle detected') as exc_info:
        PassManager().run(first_pass)

    assert exc_info.value.passrun[0].proto is first_pass.proto


def test_runner_cycle_across_three_passes_can_use_fallback():
    """Verify that a FAILED hook can recover a multi-pass cycle."""

    @analysis(on=Forest)
    def first_pass(connector, forest):
        return {'result': connector.gather(second_pass)}

    @analysis(on=Forest)
    def second_pass(connector, forest):
        return {'result': connector.gather(third_pass)}

    @analysis(on=Forest)
    def third_pass(connector, forest):
        return {'result': connector.gather(first_pass)}

    @analysis(on=Forest)
    def fallback_pass(connector, forest):
        return {'result': 'fallback'}

    cycle_failures = []

    def recover_cycle(failed):
        if isinstance(failed.failure, PassCycleError):
            cycle_failures.append(failed.failure)
            return failed.connector.run(fallback_pass)
        return None

    pm = PassManager()
    pm.hooks.install(
        recover_cycle, when='failed', kind='analysis', level='forest'
    )

    completed = pm.run(first_pass)

    assert completed.result == 'fallback'
    assert len(cycle_failures) == 1
    assert cycle_failures[0].passrun[0].proto is first_pass.proto
    assert pm._ctx._metadata_stack == {}


def test_runner_cycle_failure_hook_can_run_a_fallback_pass():
    """Verify that a failed hook can recover a cycle by returning a fallback pass."""

    @analysis(on=Forest)
    def recursive_pass(connector, forest):
        return {'result': connector.gather(recursive_pass)}

    fallback_calls = []

    @analysis(on=Forest)
    def fallback_pass(connector, forest):
        fallback_calls.append(forest)
        return {'result': 'fallback'}

    cycle_failures = []

    def recover_cycle(failed):
        if isinstance(failed.failure, PassCycleError):
            cycle_failures.append(failed.failure)
            return failed.connector.run(fallback_pass)
        return None

    pm = PassManager()
    pm.hooks.install(
        recover_cycle, when='failed', kind='analysis', level='forest'
    )

    completed = pm.run(recursive_pass)

    assert completed.result == 'fallback'
    assert completed.passe.proto is fallback_pass.proto
    assert len(cycle_failures) == 1
    assert cycle_failures[0].passrun[0].proto is recursive_pass.proto
    assert len(fallback_calls) == 1
    assert pm._ctx._metadata_stack == {}

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise ValueError('not a cycle')

    with pytest.raises(ValueError, match='not a cycle'):
        pm.run(failing_pass)

    assert len(fallback_calls) == 1
    assert pm._ctx._metadata_stack == {}


def test_pass_context_accumulates_nested_used_paths():
    """Verify that nested runs accumulate the paths used by a pass."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    forest = Forest()
    tree = _tree()
    outer = (inspect_node(), (forest, tree, tree.root), {})
    nested = (inspect_forest(), (forest,), {})
    context = PassContext()

    with context._push_pass(outer) as metadata:
        with context._push_pass(nested):
            assert context._current_passrun == nested

    assert metadata.used_paths == frozenset({outer[1], nested[1]})


def test_pass_context_accumulates_paths_for_nested_runs_at_different_levels():
    """Verify that nested runs at different levels contribute their paths."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    forest = Forest()
    tree = _tree()
    outer = (inspect_forest(), (forest,), {})
    nested = (inspect_node(), (forest, tree, tree.root), {})
    context = PassContext()

    with context._push_pass(outer) as metadata:
        with context._push_pass(nested):
            pass

    assert metadata.used_paths == frozenset({outer[1], nested[1]})


def test_pass_context_rejects_a_confused_debug_stack():
    """Verify that out-of-order internal stack changes are detected in debug mode."""

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    forest = Forest()
    first = (inspect_node(), (forest,), {})
    second = (inspect_node(), (forest, _tree()), FrozenDict(other=True))
    context = PassContext()
    pushed = context._push_pass(first)
    pushed.__enter__()
    context._metadata_stack[second] = {second[1]}

    with pytest.raises(RuntimeError, match='pass context is confused'):
        pushed.__exit__(None, None, None)


def test_manager_configures_pass_aliases():
    """Verify that a configured pass is retrievable under its alias."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': len(forest)}

    pm = PassManager()
    pm.configure(inspect_forest, 'forest_info')

    assert pm.get_passe('forest_info').name == 'forest_info'


def test_manager_rejects_unknown_pass_aliases():
    """Verify that retrieving an unknown pass alias raises KeyError."""
    with pytest.raises(KeyError, match='unknown'):
        PassManager().get_passe('unknown')


def test_manager_run_rejects_more_than_two_elements():
    """Verify that pass execution accepts at most two target elements."""

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    with pytest.raises(TypeError, match='at most 3 positional arguments'):
        PassManager().run(inspect_node, 'one', 'two', 'three')


def test_manager_requires_a_target_for_node_passes():
    """Verify that node passes require at least one target element."""

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    with pytest.raises(TypeError, match='NODE pass expect at least one'):
        PassManager().run(inspect_node)


def test_manager_requires_exactly_one_target_for_tree_passes():
    """Verify that tree passes reject zero or multiple target elements."""

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        return {'result': None}

    with pytest.raises(TypeError, match='TREE pass expect exactly one'):
        PassManager().run(inspect_tree)

    with pytest.raises(TypeError, match='TREE pass expect exactly one'):
        PassManager().run(inspect_tree, 'one', 'two')


def test_manager_rejects_targets_for_forest_passes():
    """Verify that forest passes reject explicit target elements."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    with pytest.raises(TypeError, match='FOREST pass expect exactly zero'):
        PassManager().run(inspect_forest, 'mod')


def test_manager_runs_forest_analysis_and_passes_run_options_to_hooks():
    """Verify that forest analysis returns its result and preserves run options."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': len(forest)}

    pm = PassManager()
    received = []
    pm.hooks.install(
        lambda obj: received.append(obj.run_options),
        when='unprepared',
        kind='analysis',
        level='forest',
    )

    completed = pm.run(inspect_forest, trigger='test')

    assert completed.result == 0
    assert received == [{'trigger': 'test'}]


def test_manager_rejects_unhashable_run_options_before_running_pass():
    """Verify that unhashable run options fail before the pass body executes."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest, *, selector):
        calls.append(selector)
        return {'result': None}

    with pytest.raises(TypeError, match='pass run options must be hashable'):
        PassManager().run(inspect_forest, selector=('a', ['b']))

    assert calls == []


def test_manager_runs_tree_analysis_by_identifier_root_and_tree():
    """Verify that tree analyses accept each documented way to identify a tree."""

    @analysis(on=Tree)
    def identify_tree(connector, tree):
        return {'result': tree.identifier}

    pm, tree = _manager_with_tree()

    assert pm.gather(identify_tree, tree.identifier) == 'mod'
    assert pm.gather(identify_tree, tree.root) == 'mod'
    assert pm.gather(identify_tree, tree) == 'mod'


def test_manager_promotes_single_tree_argument_to_node_root():
    """Verify that a node pass given one tree target runs on that tree's root."""

    @analysis(on=ast.Module)
    def inspect_module(connector, node):
        return {'result': node}

    pm, tree = _manager_with_tree()

    assert pm.gather(inspect_module, tree) is tree.root


def test_manager_does_not_automatically_add_an_unregistered_tree_argument():
    """Verify that targeting an unregistered Tree raises an error."""

    @analysis(on=Tree)
    def identify_tree(connector, tree):
        return {'result': tree.identifier}

    pm = PassManager()
    tree = _tree('new_tree')

    with pytest.raises(ValueError, match='not in the passmanager'):
        pm.gather(identify_tree, tree)


def test_manager_runs_node_analysis_on_a_specific_node():
    """Verify that a node pass can target an explicit node inside a tree."""

    @analysis(on=ast.Name)
    def inspect_name(connector, node):
        return {'result': node.id}

    pm, tree = _manager_with_tree()
    name = tree.root.body[0].targets[0]

    assert pm.gather(inspect_name, tree.identifier, name) == 'x'


def test_manager_can_add_an_ast_with_metadata():
    """Verify that adding an AST creates and stores a tree with metadata."""
    pm = PassManager()
    root = ast.parse('x = 1')

    pm.add(root, 'mod', filename='mod.py')

    assert pm.trees['mod'].root is root
    assert pm.trees['mod'].attributes['filename'] == 'mod.py'


def test_manager_requires_identifier_when_adding_an_ast():
    """Verify that adding a root AST without an identifier raises TypeError."""
    with pytest.raises(TypeError, match='identifier.*required'):
        PassManager().add(ast.parse('x = 1'))


def test_manager_rejects_identifier_when_adding_a_tree():
    """Verify that a Tree instance cannot be accompanied by a separate identifier."""
    with pytest.raises(TypeError, match='identifier.*not supported'):
        PassManager().add(_tree(), 'other')


def test_manager_rejects_metadata_when_adding_a_tree():
    """Verify that a Tree instance cannot be accompanied by separate metadata."""
    with pytest.raises(TypeError, match='keywords not supported'):
        PassManager().add(_tree(), filename='other.py')


def test_manager_add_is_idempotent_for_an_existing_tree():
    """Verify that adding the same Tree instance again leaves the forest unchanged."""
    pm, tree = _manager_with_tree()

    pm.add(tree)

    assert len(pm.trees) == 1


@pytest.mark.parametrize('target', ['identifier', 'root', 'tree'])
def test_manager_removes_tree_by_supported_target(target):
    """Verify that a tree can be removed by identifier, root, or Tree object."""
    pm, tree = _manager_with_tree()
    target_value = {
        'identifier': tree.identifier,
        'root': tree.root,
        'tree': tree,
    }[target]

    pm.remove(target_value)

    assert len(pm.trees) == 0


def test_manager_remove_is_noop_for_unknown_tree():
    """Verify that removing a tree absent from the forest does not change it."""
    pm = PassManager()

    pm.remove('missing')

    assert len(pm.trees) == 0


def test_manager_gather_rejects_transformation_passes():
    """Verify that gather accepts analyses only."""

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': False}

    with pytest.raises(TypeError):
        PassManager().gather(transform_forest)


def test_manager_apply_rejects_analysis_passes():
    """Verify that apply accepts transformations only."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    with pytest.raises(TypeError):
        PassManager().apply(inspect_forest)


def test_manager_runs_passes_by_configured_alias():
    """Verify that string aliases dispatch to their configured pass."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': len(forest)}

    pm = PassManager()
    pm.configure(inspect_forest, 'count')

    assert pm.gather('count') == 0


def test_manager_apply_runs_a_configured_transformation_alias():
    """Verify that apply resolves a configured transformation."""

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': True}

    pm = PassManager()
    pm.configure(transform_forest, 'transform')

    assert pm.apply('transform') is True


def test_manager_run_resolves_a_configured_pass_alias():
    """Verify that run resolves a configured pass name."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 10}

    pm = PassManager()
    pm.configure(inspect_forest, 'inspect')

    assert pm.run('inspect').result == 10


def test_manager_selects_runner_from_pass_kind():
    """Verify that analysis and transformation passes use their matching runners."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': False}

    pm = PassManager()

    assert isinstance(pm._get_runner(inspect_forest(), (pm.trees,), {}), AnalysisRunner)
    assert isinstance(pm._get_runner(transform_forest(), (pm.trees,), {}), TransformationRunner)


def test_analysis_runner_builds_completed_result_and_attributes():
    """Verify that analysis completion separates result data from other attributes."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 42, 'complete': True}

    completed = PassManager().run(inspect_forest)

    assert completed.result == 42
    assert completed.update is False
    assert completed.attributes == {'complete': True}


def test_transformation_runner_builds_completed_update_and_attributes():
    """Verify that transformation completion separates its update flag from attributes."""

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': True, 'preserved': ()}

    completed = PassManager().run(transform_forest)

    assert completed.update is True
    assert completed.result is None
    assert completed.attributes == {'preserved': ()}


def test_runner_converts_iterable_pass_results_to_a_dict():
    """Verify that non-dictionary pass results are converted from key-value pairs."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return iter((('result', 7), ('complete', True)))

    completed = PassManager().run(inspect_forest)

    assert completed.result == 7
    assert completed.attributes == {'complete': True}


def test_runner_prepares_node_pass_rejecting_missing_element():
    """Verify that direct runner preparation rejects a node pass without an element."""

    @analysis(on=ast.Module)
    def inspect_node(connector, node):
        return {'result': None}

    pm = PassManager()
    unprepared = UnpreparedPass(inspect_node(), (pm.trees,), pm, {})

    with pytest.raises(AssertionError):
        Runner.prepare(unprepared)


def test_runner_prepares_node_pass_rejecting_wrong_runtime_type():
    """Verify that direct runner preparation validates a node pass's runtime type."""

    @analysis(on=ast.Module)
    def inspect_module(connector, node):
        return {'result': None}

    pm, tree = _manager_with_tree()
    unprepared = UnpreparedPass(
        inspect_module(), (pm.trees, tree, ast.Constant(value=1)), pm, {}
    )

    with pytest.raises(TypeError, match='unexpected type'):
        Runner.prepare(unprepared)


def test_runner_converts_analysis_dependencies_to_lazy_descriptors(monkeypatch):
    """Verify that analysis dependencies are gathered lazily when accessed."""

    @analysis(on=ast.Module)
    def dependency(connector, node):
        return {'result': 'dependency'}

    @analysis(on=ast.Module, dependencies=(dependency,))
    def inspect_module(connector, node):
        return {'result': connector.deps.dependency}

    pm, tree = _manager_with_tree()
    calls = []

    def gather(*args, **kwargs):
        calls.append((args, kwargs))
        return 'dependency'

    monkeypatch.setattr(pm, 'gather', gather)
    prepared = Runner.prepare(
        UnpreparedPass(inspect_module(), (pm.trees, tree, tree.root), pm, {})
    )

    assert calls == []
    assert prepared.connector.deps.dependency == 'dependency'
    assert len(calls) == 1


def test_runner_gathers_analysis_dependencies_eagerly_for_transformations(monkeypatch):
    """Verify that transformations evaluate their analysis dependencies during preparation."""

    @analysis(on=ast.Module)
    def dependency(connector, node):
        return {'result': 'dependency'}

    @transformation(on=ast.Module, dependencies=(dependency,))
    def transform(connector, node):
        return {'update': False}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'gather', lambda *args, **kwargs: calls.append(kwargs) or 1)

    prepared = Runner.prepare(
        UnpreparedPass(transform(), (pm.trees, tree, tree.root), pm, {})
    )

    assert calls == [{'cache_only': False}]
    assert prepared.connector.deps.dependency == 1


def test_runner_requests_transformation_dependencies_from_analyses_cache_only(monkeypatch):
    """Verify that analysis transformation dependencies are requested cache-only."""

    @transformation(on=ast.Module)
    def dependency(connector, node):
        return {'update': False}

    @analysis(on=ast.Module, dependencies=(dependency,))
    def inspect_module(connector, node):
        return {'result': None}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'apply', lambda *args, **kwargs: calls.append((args, kwargs)))

    Runner.prepare(
        UnpreparedPass(inspect_module(), (pm.trees, tree, tree.root), pm, {})
    )

    assert calls == [((dependency, tree, tree.root), {'cache_only': True})]


@pytest.mark.parametrize('operation', ['add', 'remove'])
@pytest.mark.parametrize(
    ('level', 'on'),
    [
        ('node', ast.Name),
        ('tree', Tree),
        ('forest', Forest),
    ],
)
def test_builtin_forest_transformations_cannot_be_analysis_dependencies(
    operation, level, on
):
    """Verify builtin forest changes cannot run as uncached analysis dependencies."""
    pm = PassManager()
    target = _tree('target')
    member = _tree('member')
    pm.add(target)
    if operation == 'remove':
        pm.add(member)

    dependency = (
        _add_tree(tree=member)
        if operation == 'add'
        else _remove_tree(tree=member)
    )
    observed = []
    @analysis(on=on, dependencies=(dependency,))
    def dependent(connector, element):
        observed.append(member in pm.trees)
        return {'result': member in pm.trees}

    run_args = {
        'node': (target, target.root.body[0].targets[0]),
        'tree': (target,),
        'forest': (),
    }[level]
    with pytest.raises(TypeError, match='analysis cannot depend on transformation if you are not using the cache plugin'):
        pm.gather(dependent, *run_args)

    assert observed == []
    assert (member in pm.trees) is (operation == 'remove')


@pytest.mark.parametrize('operation', ['add', 'remove'])
@pytest.mark.parametrize(
    ('level', 'on', 'allowed'),
    [
        ('node', ast.Name, False),
        ('tree', Tree, False),
        ('forest', Forest, True),
    ],
)
def test_builtin_forest_transformations_can_be_transformation_dependencies(
    operation, level, on, allowed
):
    """Verify builtin forest changes as dependencies of transformations across levels."""
    pm = PassManager()
    target = _tree('target')
    member = _tree('member')
    pm.add(target)
    if operation == 'remove':
        pm.add(member)

    dependency = (
        _add_tree(tree=member)
        if operation == 'add'
        else _remove_tree(tree=member)
    )
    observed = []
    @transformation(on=on, dependencies=(dependency,))
    def dependent(connector, element):
        observed.append(member in pm.trees)
        return {'update': False}

    run_args = {
        'node': (target, target.root.body[0].targets[0]),
        'tree': (target,),
        'forest': (),
    }[level]

    if not allowed:
        with pytest.raises(TypeError, match='cannot depend on enclosing level'):
            pm.apply(dependent, *run_args)
        assert observed == []
        assert (member in pm.trees) is (operation == 'remove')
        return

    result = pm.apply(dependent, *run_args)

    assert observed == [operation == 'add']
    assert result is False
    assert (member in pm.trees) is (operation == 'add')


def test_runner_uses_cache_only_for_node_transformation_dependency_from_tree_analysis(monkeypatch):
    """Verify that a tree analysis requests node transformation dependencies cache-only."""

    @transformation(on=ast.Module)
    def transform_module(connector, node):
        return {'update': False}

    @analysis(on=Tree, dependencies=(transform_module,))
    def inspect_tree(connector, tree):
        return {'result': None}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'apply', lambda *args, **kwargs: calls.append((args, kwargs)))

    Runner.prepare(UnpreparedPass(inspect_tree(), (pm.trees, tree), pm, {}))

    assert calls == [((transform_module, tree), {'cache_only': True})]


def test_runner_uses_cache_only_for_enclosing_transformation_dependency_from_analysis(monkeypatch):
    """Verify node analyses request enclosing transformation dependencies cache-only."""

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': False}

    @analysis(on=ast.Module, dependencies=(transform_forest,))
    def inspect_module(connector, node):
        return {'result': None}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'apply', lambda *args, **kwargs: calls.append((args, kwargs)))

    Runner.prepare(
        UnpreparedPass(inspect_module(), (pm.trees, tree, tree.root), pm, {})
    )

    assert calls == [((transform_forest,), {'cache_only': True})]


def test_runner_rejects_lower_level_transformation_dependency_from_forest():
    """Verify that a forest analysis cannot depend on a node transformation."""

    @transformation(on=ast.Module)
    def transform_module(connector, node):
        return {'update': False}

    @analysis(on=Forest, dependencies=(transform_module,))
    def inspect_forest(connector, forest):
        return {'result': None}

    pm = PassManager()

    with pytest.raises(TypeError, match='compatible or enclosing level'):
        Runner.prepare(UnpreparedPass(inspect_forest(), (pm.trees,), pm, {}))


def test_runner_requires_cached_transitive_transformation_dependency_from_analysis():
    """Verify that transitive transformations of an analysis must already be cached."""

    @transformation(on=ast.Module)
    def base_transform(connector, node):
        return {'update': False}

    @transformation(on=ast.Module, dependencies=(base_transform,))
    def outer_transform(connector, node):
        return {'update': False}

    @analysis(on=ast.Module, dependencies=(outer_transform,))
    def inspect_module(connector, node):
        return {'result': None}

    pm, tree = _manager_with_tree()

    with pytest.raises(TypeError, match='analysis cannot depend on transformation if you are not using the cache plugin'):
        pm.gather(inspect_module, tree)


@pytest.mark.parametrize('method', ['apply', 'run'])
def test_analysis_connector_cannot_run_transformations(method):
    """Verify that connector apply and run reject transformations from analyses."""

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        return {'update': True}

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        getattr(connector, method)(transform_tree, tree)
        return {'result': None}

    pm, tree = _manager_with_tree()

    with pytest.raises(TypeError, match='analyses cannot run transformations'):
        pm.gather(inspect_tree, tree)


@pytest.mark.parametrize('method', ['apply', 'run'])
def test_analysis_cannot_call_pass_manager_transform_methods(method):
    """Verify that direct pass-manager apply and run calls reject transformations."""

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        return {'update': True}

    pm, tree = _manager_with_tree()

    @analysis(on=Tree)
    def inspect_tree(connector, target):
        getattr(pm, method)(transform_tree, target)
        return {'result': None}

    with pytest.raises(TypeError, match='analyses cannot run transformations'):
        pm.gather(inspect_tree, tree)


def test_runner_promotes_node_analysis_dependency_to_tree_target(monkeypatch):
    """Verify that a tree analysis can lazily gather a node analysis on the tree root."""

    @analysis(on=ast.Module)
    def inspect_module(connector, node):
        return {'result': None}

    @analysis(on=Tree, dependencies=(inspect_module,))
    def inspect_tree(connector, tree):
        return {'result': None}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'gather', lambda *args, **kwargs: calls.append(args) or None)
    prepared = Runner.prepare(UnpreparedPass(inspect_tree(), (pm.trees, tree), pm, {}))

    prepared.connector.deps.inspect_module

    assert calls == [(inspect_module, tree)]


def test_runner_rejects_enclosing_transformation_dependency_for_transformation():
    """Verify that a node transformation cannot depend on an enclosing forest transformation."""

    @transformation(on=Forest)
    def transform_forest(connector, forest):
        return {'update': False}

    @transformation(on=ast.Module, dependencies=(transform_forest,))
    def transform_module(connector, node):
        return {'update': False}

    pm, tree = _manager_with_tree()

    with pytest.raises(TypeError, match='cannot depend on enclosing level'):
        Runner.prepare(
            UnpreparedPass(transform_module(), (pm.trees, tree, tree.root), pm, {})
        )


def test_runner_rejects_analysis_dependency_with_missing_parameters():
    """Verify that an analysis cannot declare a dependency missing required arguments."""

    @analysis(on=ast.Module)
    def dependency(connector, node, *, required):
        return {'result': None}

    @analysis(on=ast.Module, dependencies=(dependency(),))
    def inspect_module(connector, node):
        return {'result': None}

    pm, tree = _manager_with_tree()

    with pytest.raises(TypeError, match='missing a required parameter'):
        Runner.prepare(
            UnpreparedPass(inspect_module(), (pm.trees, tree, tree.root), pm, {})
        )


def test_runner_rejects_lower_level_analysis_dependency_from_forest():
    """Verify that a forest analysis cannot directly depend on a node analysis."""

    @analysis(on=ast.Module)
    def inspect_module(connector, node):
        return {'result': None}

    @analysis(on=Forest, dependencies=(inspect_module,))
    def inspect_forest(connector, forest):
        return {'result': None}

    pm = PassManager()

    with pytest.raises(TypeError, match='compatible or enclosing level'):
        Runner.prepare(UnpreparedPass(inspect_forest(), (pm.trees,), pm, {}))


def test_runner_uses_cache_only_for_outer_scope_analysis_of_transformations(monkeypatch):
    """Verify that a transformation's enclosing analysis dependency is requested cache-only."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': None}

    @transformation(on=ast.Module, dependencies=(inspect_forest,))
    def transform_module(connector, node):
        return {'update': False}

    pm, tree = _manager_with_tree()
    calls = []
    monkeypatch.setattr(pm, 'gather', lambda *args, **kwargs: calls.append(kwargs) or None)

    Runner.prepare(
        UnpreparedPass(transform_module(), (pm.trees, tree, tree.root), pm, {})
    )

    assert calls == [{'cache_only': True}]


def test_connector_analysis_dependency_is_lazy_and_repeated_access_recomputes(monkeypatch):
    """Verify that a lazy dependency callback is invoked on each attribute access."""

    @analysis(on=Forest)
    def dependency(connector, forest):
        return {'result': None}

    @analysis(on=Forest, dependencies=(dependency,))
    def inspect_forest(connector, forest):
        return {'result': None}

    pm = PassManager()
    calls = []
    monkeypatch.setattr(pm, 'gather', lambda *args, **kwargs: calls.append(1) or len(calls))
    prepared = Runner.prepare(
        UnpreparedPass(inspect_forest(), (pm.trees,), pm, {})
    )

    first = prepared.connector.deps.dependency
    second = prepared.connector.deps.dependency

    assert (first, second) == (1, 2)


def test_runner_executes_unprepared_hook_short_circuit():
    """Verify that an unprepared hook can return an existing completed result immediately."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 1}

    pm = PassManager()
    completed = pm.run(inspect_forest)
    pm.hooks.install(
        lambda obj: completed,
        when='unprepared',
        kind='analysis',
        level='forest',
    )

    assert pm.run(inspect_forest) is completed


def test_runner_rejects_unsupported_unprepared_hook_result():
    """Verify that an unprepared hook returning an unsupported object raises AssertionError."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 1}

    pm = PassManager()
    pm.hooks.install(
        lambda obj: object(),
        when='unprepared',
        kind='analysis',
        level='forest',
    )

    with pytest.raises(AssertionError, match='unsupported object type'):
        pm.run(inspect_forest)


def test_runner_rejects_unsupported_failed_hook_result():
    """Verify that a failed hook returning an unsupported object raises AssertionError."""

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise RuntimeError('failed')

    pm = PassManager()
    pm.hooks.install(
        lambda obj: object(),
        when='failed',
        kind='analysis',
        level='forest',
    )

    with pytest.raises(AssertionError, match='unsupported object type'):
        pm.run(failing_pass)


def test_runner_reraises_failure_from_failed_pass():
    """Verify that an unmodified failed pass re-raises its original exception."""

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise ValueError('original failure')

    with pytest.raises(ValueError, match='original failure'):
        PassManager().run(failing_pass)


def test_runner_cleans_context_after_pass_failure():
    """Verify that a failed pass leaves no active context and does not block later runs."""

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise ValueError('failure')

    @analysis(on=Forest)
    def succeeding_pass(connector, forest):
        return {'result': 'ok'}

    pm = PassManager()
    with pytest.raises(ValueError, match='failure'):
        pm.run(failing_pass)

    assert pm._ctx._metadata_stack == {}
    assert pm.run(succeeding_pass).result == 'ok'


def test_runner_failed_hook_can_replace_failure():
    """Verify that a failed hook can replace the exception raised by a pass."""

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise ValueError('original failure')

    def replace_failure(failed):
        return FailedPass(
            failed.passe,
            failed.pointer,
            failed.connector,
            failed.run_options,
            RuntimeError('replacement failure'),
        )

    pm = PassManager()
    pm.hooks.install(
        replace_failure, when='failed', kind='analysis', level='forest'
    )

    with pytest.raises(RuntimeError, match='replacement failure'):
        pm.run(failing_pass)


def test_runner_failed_hook_can_return_completed_pass():
    """Verify that a failed hook may recover by returning a completed pass."""

    @analysis(on=Forest)
    def failing_pass(connector, forest):
        raise ValueError('failed')

    pm = PassManager()
    recovered = pm.run(analysis(on=Forest)(lambda connector, forest: {'result': 10}))
    pm.hooks.install(
        lambda obj: recovered,
        when='failed',
        kind='analysis',
        level='forest',
    )

    assert pm.run(failing_pass) is recovered


def test_plugin_registration_installs_nested_plugins():
    """Verify that a plugin can register itself and a nested plugin on the manager."""

    class Nested:
        name = 'nested_plugin'
        def register(self, registrar):
            yield self

    class Plugin:
        name = 'plugin'

        def register(self, registrar):
            return [self, *Nested().register(registrar)]

    pm = PassManager([Plugin])

    assert pm.plugin.name == 'plugin'
    assert pm.nested_plugin.name == 'nested_plugin'


def test_plugin_registration_rejects_existing_manager_attribute():
    """Verify that a plugin cannot overwrite an existing PassManager attribute."""

    class Plugin:
        name = 'hooks'

        def register(self, registrar):
            return [self]

    with pytest.raises(ValueError, match='already has an attribute'):
        PassManager([Plugin])


def test_hooks_cast_enum_names_to_selectors():
    """Verify that hook selector names normalize to their declared enum values."""
    selectors = Hooks._cast_string_values('COMPLETED', 'ANALYSIS', 'TREE')

    assert selectors == (
        Trigger.COMPLETED,
        PassKind.ANALYSIS,
        Level.TREE,
    )


def test_hooks_cast_enum_and_integer_selectors():
    """Verify that hook selector conversion preserves enum and integer values."""
    selectors = Hooks._cast_string_values(
        Trigger.PREPARED,
        PassKind.TRANSFORMATION,
        Level.NODE,
    )

    assert selectors == (
        Trigger.PREPARED,
        PassKind.TRANSFORMATION,
        Level.NODE,
    )


@pytest.mark.parametrize(
    'selectors',
    [
        ('invalid', 'analysis', 'tree'),
        ('completed', 'invalid', 'tree'),
        ('completed', 'analysis', 'invalid'),
    ],
)
def test_hooks_reject_unknown_selector_names(selectors):
    """Verify that hook selector conversion rejects unknown names at every selector position."""
    with pytest.raises(KeyError):
        Hooks._cast_string_values(*selectors)


def test_hooks_wildcards_register_all_selector_combinations():
    """Verify that wildcards expand across pass kinds and execution levels."""
    hooks = Hooks()
    callback = lambda obj: None

    hooks.install(
        callback,
        when='completed',
        kind='all',
        level='all',
    )

    registered = [
        hook
        for kind in PassKind
        for level in Level
        for hook in hooks.get(Trigger.COMPLETED, kind, level)
    ]
    assert len(registered) == len(PassKind) * len(Level)
    assert all(hook is callback for hook in registered)


def test_hooks_get_returns_hooks_for_exact_selectors():
    """Verify that hook lookup matches the trigger, kind, and execution level."""
    hooks = Hooks()
    callback = lambda obj: None
    hooks.install(
        callback,
        when=Trigger.COMPLETED,
        kind=PassKind.ANALYSIS,
        level=Level.TREE,
    )

    assert list(
        hooks.get(
            Trigger.COMPLETED,
            PassKind.ANALYSIS,
            Level.TREE,
        )
    ) == [callback]


def test_hooks_uninstall_removes_all_registrations_for_callback():
    """Verify that uninstall removes each registration associated with a hook callback."""
    hooks = Hooks()
    callback = lambda obj: None
    hooks.install(
        callback,
        when='completed',
        kind='all',
        level='tree',
    )

    hooks.uninstall(callback)

    assert all(
        not list(hooks.get(Trigger.COMPLETED, kind, Level.TREE))
        for kind in PassKind
    )


def test_runner_hook_chain_keeps_object_when_hook_returns_none():
    """Verify that a hook returning None leaves the current pass object unchanged."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 1}

    pm = PassManager()
    seen = []
    pm.hooks.install(
        lambda obj: seen.append(obj) or None,
        when='prepared',
        kind='analysis',
        level='forest',
    )

    completed = pm.run(inspect_forest)

    assert isinstance(seen[0], PreparedPass)
    assert completed.result == 1


def test_runner_hook_chain_uses_replacement_object():
    """Verify that a hook can replace the prepared pass object."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 1}

    pm = PassManager()
    replacements = []

    def replace_options(prepared):
        replacement = PreparedPass(
            prepared.passe,
            prepared.pointer,
            prepared.connector,
            {'replacement': True},
        )
        replacements.append(replacement)
        return replacement

    pm.hooks.install(
        replace_options,
        when='prepared',
        kind='analysis',
        level='forest',
    )

    completed = pm.run(inspect_forest)

    assert completed.run_options == {'replacement': True}
    assert replacements


def test_runner_completed_hook_receives_result_by_execution_level():
    """Verify that completed hooks run for the pass's declared execution level."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': 1}

    pm = PassManager()
    seen = []
    pm.hooks.install(
        lambda completed: seen.append(completed.meta.used_paths),
        when='completed',
        kind='analysis',
        level='forest',
    )

    completed = pm.run(inspect_forest)

    assert seen == [frozenset({completed.pointer})]


def test_analysis_results_are_complete_by_default():
    """Verify that analysis results remain complete unless changed by a future API."""

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': len(forest)}

    completed = PassManager().run(inspect_forest)

    assert completed.attributes['complete'] is True