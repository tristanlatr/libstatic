"""Tests for pass cache primitives and revision tracking."""

import ast

import attrs
import pytest

from libstatic.passmanager._caching import (
    CacheMissError,
    PassManagerCache,
    PassPattern,
    PreservedAnalyses,
    RevisionTracker,
)
from libstatic.passmanager._model import Forest, Level, Tree
from libstatic.passmanager._passe import analysis, transformation
from libstatic.passmanager._passmanager import PassManager


def _cache():
    return PassManagerCache()


def _analysis_pass():
    @analysis(on=ast.Module)
    def inspect_module(connector, node, *, mode='default', required):
        return {'result': (mode, required)}

    return inspect_module


def _completed(passe=None, *, target='forest', pm=None, **options):
    pm = pm or PassManager()
    if passe is None:
        if target == 'tree':
            passe = analysis(on=Tree)(
                lambda connector, tree: {'result': tree.identifier}
            )
        else:
            passe = analysis(on=Forest)(
                lambda connector, forest: {'result': 'value'}
            )
    if target == 'tree':
        tree = Tree(ast.parse('x = 1'), 'mod')
        pm.add(tree)
        return pm.run(passe, tree, **options)
    return pm.run(passe, **options)


def test_pattern_rejects_non_pass_instances():
    """Verify that a pass pattern never matches a prototype or unrelated object."""
    passe = _analysis_pass()
    pattern = PassPattern.like(passe)

    assert not pattern.matches(passe)
    assert not pattern.matches(object())


def test_pattern_requires_the_same_prototype():
    """Verify that a pattern only matches instances derived from its prototype."""
    passe = _analysis_pass()
    other = _analysis_pass()

    assert not PassPattern.like(passe).matches(other(required='x'))


def test_pattern_matches_an_instance_when_all_predicates_pass():
    """Verify that matching predicates accept a pass instance with matching arguments."""
    passe = _analysis_pass()
    pattern = PassPattern.like(
        passe,
        mode=lambda value: value == 'default',
        required=lambda value: value.startswith('ok'),
    )

    assert pattern.matches(passe(required='okay'))


def test_pattern_rejects_an_instance_when_a_predicate_fails():
    """Verify that a false predicate prevents a pass instance from matching."""
    passe = _analysis_pass()
    pattern = PassPattern.like(passe, required=lambda value: value == 'expected')

    assert not pattern.matches(passe(required='other'))


def test_pattern_ignores_predicates_for_missing_required_parameters():
    """Verify that an unbound required parameter does not prevent pattern matching."""
    passe = _analysis_pass()
    pattern = PassPattern.like(passe, required=lambda value: False)

    assert pattern.matches(passe())


def test_pattern_rejects_unknown_parameter_names():
    """Verify that preservation patterns cannot refer to undeclared pass parameters."""
    passe = _analysis_pass()

    with pytest.raises(ValueError, match='unknown pass parameter'):
        PassPattern.like(passe, missing=lambda value: True)


def test_pattern_rejects_non_callable_predicates():
    """Verify that preservation predicates must be callable."""
    passe = _analysis_pass()

    with pytest.raises(TypeError, match='predicates must be callable'):
        PassPattern.like(passe, required=True)


def test_pattern_accepts_a_pass_prototype():
    """Verify that a pattern created from an instance matches its prototype's derivations."""
    passe = _analysis_pass()
    pattern = PassPattern.like(
        passe(required='first').proto,
        required=lambda value: value == 'second',
    )

    assert pattern.matches(passe(required='second'))


def test_pattern_rejects_an_unsupported_pass_reference():
    """Verify that pattern references must be prototypes, instances, or aliases."""
    with pytest.raises(TypeError, match='expected a pass prototype or alias'):
        PassPattern.like(object())


def test_pattern_alias_must_be_resolved_before_matching():
    """Verify that alias-based patterns are resolved by the pass manager before use."""
    pattern = PassPattern.like('inspect')

    with pytest.raises(ValueError, match='aliases must be resolved'):
        pattern.matches(_analysis_pass()())


def test_pattern_matches_declared_optional_default_values():
    """Verify that predicates are evaluated against defaults when optional arguments are absent."""
    passe = _analysis_pass()
    pattern = PassPattern.like(passe, mode=lambda value: value == 'default')

    assert pattern.matches(passe(required='set'))


def test_preserved_analyses_accepts_prototypes_instances_aliases_and_patterns():
    """Verify that each supported preservation input is stored and matched correctly."""
    passe = _analysis_pass()
    configured = passe._replace(name='inspect')
    instance = configured(required='specific')
    pattern = PassPattern.like(configured, required=lambda value: value == 'pattern')
    preserved = PreservedAnalyses(
        {'inspect': configured}.__getitem__,
        [configured, instance, 'inspect', pattern],
    )

    assert configured() in preserved
    assert instance in preserved
    assert configured(required='pattern') in preserved


def test_preserved_analyses_does_not_match_an_unpreserved_instance():
    """Verify that analyses with no matching prototype, instance, or pattern are absent."""
    passe = _analysis_pass()
    preserved = PreservedAnalyses(lambda name: passe, [passe(required='one')])

    assert passe(required='two') not in preserved


def test_preserved_analyses_rejects_unsupported_values():
    """Verify that unsupported preservation declarations raise a clear TypeError."""
    with pytest.raises(TypeError, match='expected a pass prototype'):
        PreservedAnalyses(lambda name: _analysis_pass(), [object()])


def test_cache_returns_none_for_a_missing_run():
    """Verify that cache lookup returns None when no result has been stored."""
    passe = _analysis_pass()
    cache = _cache()

    assert cache.get((passe(), (Forest(),), {})) is None


def test_cache_stores_and_returns_a_result_at_the_same_revision():
    """Verify that a result is returned while its target revision remains unchanged."""
    completed = _completed()
    cache = _cache()

    cache.set(completed)

    assert cache.get((completed.passe, completed.pointer, {})) is completed


def test_cache_snapshot_uses_dependencies_without_a_tree_target():
    """Verify that tree dependencies remain tracked when a result has no tree target."""
    completed = _completed(target='tree')
    result = attrs.evolve(completed, pointer=(completed.pointer[0],))
    cache = _cache()

    snapshot = cache._revision_snapshot(result)

    assert snapshot == (None, ((completed.pointer[1], 0),))


def test_cache_snapshot_tracks_forest_and_tree_paths_independently():
    """Verify that a nested forest dependency retains its own membership path."""
    pm = PassManager()
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        return {'result': len(forest)}

    @analysis(on=Tree)
    def inspect_tree(connector, target):
        connector.gather(inspect_forest)
        return {'result': target.identifier}

    completed = pm.run(inspect_tree, tree)
    cache = _cache()

    assert cache._revision_snapshot(completed) == (0, ((tree, 0),))


@pytest.mark.parametrize('runs_on', ['tree', 'node'])
def test_cache_snapshot_tracks_other_trees_used_by_lower_scope_analysis(runs_on):
    """Verify that edits to a non-target tree invalidate dependent tree and node results."""
    pm = PassManager()
    target_tree = Tree(ast.parse('x = 1'), 'target')
    dependency_tree = Tree(ast.parse('y = 2'), 'dependency')
    pm.add(target_tree)
    pm.add(dependency_tree)

    @analysis(on=Tree)
    def inspect_identifier(connector, tree):
        return {'result': tree.identifier}

    if runs_on == 'tree':
        @analysis(on=Tree)
        def inspect_target(connector, tree):
            connector.gather(inspect_identifier, 'dependency')
            return {'result': tree.identifier}

        target = target_tree
    else:
        @analysis(on=ast.Module)
        def inspect_target(connector, node):
            connector.gather(inspect_identifier, 'dependency')
            return {'result': node}

        target = target_tree

    @analysis(on=Tree)
    def inspect_target_tree(connector, tree):
        return {'result': tree.identifier}

    dependent_result = pm.run(inspect_target, target)
    target_result = pm.run(inspect_target_tree, target_tree)
    cache = _cache()
    cache.set(dependent_result)
    cache.set(target_result)

    snapshot = cache._revision_snapshot(dependent_result)
    assert snapshot[0] is None
    assert {tree for tree, _ in snapshot[1]} == {
        target_tree,
        dependency_tree,
    }

    cache.tracker.increment_tree_rev(dependency_tree)

    assert cache.get(
        (target_result.passe, target_result.pointer, {})
    ) is target_result
    assert cache.get(
        (dependent_result.passe, dependent_result.pointer, {})
    ) is None


def test_cache_key_rejects_unhashable_run_options():
    """Verify that cache lookups reject run options that cannot form hashable keys."""
    passe = analysis(on=Forest)(lambda connector, forest: {'result': None})
    completed = _completed(passe)
    cache = _cache()
    cache.set(completed)

    with pytest.raises(TypeError):
        cache.get(
            (completed.passe, completed.pointer, {'selector': {'names': ['a', 'b']}})
        )


def test_cache_options_accept_hashable_nested_values():
    """Verify that nested hashable options remain usable as cache keys."""
    passe = analysis(on=Forest)(lambda connector, forest: {'result': None})
    options = {'values': (('a', 'b'), frozenset({1, 2}))}
    completed = _completed(passe, **options)
    cache = _cache()
    cache.set(completed)

    assert cache.get((completed.passe, completed.pointer, options)) is completed


def test_cache_only_option_does_not_change_cache_identity():
    """Verify that cache-only control options do not distinguish cached computations."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)

    assert cache.get(
        (completed.passe, completed.pointer, {'cache_only': True})
    ) is completed


def test_cache_evicts_a_result_after_an_unpreserved_revision():
    """Verify that an unpreserved revision change removes the stale result."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.tracker.increment_forest_rev()

    assert cache.get((completed.passe, completed.pointer, {})) is None


def test_cache_evicts_entries_missing_revision_metadata():
    """Verify that an inconsistent entry without revision metadata is safely discarded."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    key = next(iter(cache.results_store))
    cache.results_revision.clear()

    assert cache._get(key) is None
    assert cache.results_store == {}


def test_cache_evicts_entries_with_invalid_revision_metadata():
    """Verify that a result with an invalid revision snapshot is treated as stale."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    key = next(iter(cache.results_store))
    cache.results_revision[key] = (-1, ())

    assert cache.get((completed.passe, completed.pointer, {})) is None


def test_cache_evicts_entries_with_malformed_revision_metadata():
    """Verify that a snapshot with the wrong shape is discarded safely."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    key = next(iter(cache.results_store))
    cache.results_revision[key] = ()

    assert cache._get(key) is None
    assert key not in cache.results_store


def test_cache_evicts_result_when_current_revision_precedes_cached_revision():
    """Verify that a revision rollback cannot make an earlier result appear current."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.tracker._forest_revision = -1

    assert cache.get((completed.passe, completed.pointer, {})) is None


def test_cache_retains_a_result_preserved_across_forest_revision():
    """Verify that declared preservation keeps a forest analysis valid across a revision."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.tracker.increment_forest_rev()
    cache.results_preservation[None] = {
        1: PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe])
    }

    assert cache.get((completed.passe, completed.pointer, {})) is completed


def test_cache_retains_a_tree_result_preserved_across_tree_revision():
    """Verify that declared preservation keeps a tree analysis valid after a tree edit."""
    completed = _completed(target='tree')
    tree = completed.pointer[1]
    cache = _cache()
    cache.set(completed)
    cache.tracker.increment_tree_rev(tree)
    preserved = PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe])
    cache.results_preservation[None] = {
        1: preserved,
    }
    cache.results_preservation[tree] = {
        1: preserved,
    }

    assert cache.get((completed.passe, completed.pointer, {})) is completed


def test_cache_updates_preserved_result_revision_after_lookup():
    """Verify that a successfully preserved result is rebased to the current revision."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    key = next(iter(cache.results_store))
    cache.tracker.increment_forest_rev()
    cache.results_preservation[None] = {
        1: PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe])
    }

    assert cache.get((completed.passe, completed.pointer, {})) is completed
    assert cache.results_revision[key] == (1, ())


def test_cache_evicts_result_when_one_of_multiple_revisions_is_not_preserved():
    """Verify that every intervening revision must preserve an analysis."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.tracker.increment_forest_rev()
    cache.tracker.increment_forest_rev()
    cache.results_preservation[None] = {
        1: PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe]),
    }

    assert cache.get((completed.passe, completed.pointer, {})) is None


def test_cache_purge_removes_stale_results():
    """Verify that purging the cache checks and removes stale result entries."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.tracker.increment_forest_rev()

    cache.purge()

    assert cache.results_store == {}


def test_cache_purge_keeps_only_forest_analyses_preserved_at_a_revision():
    """Verify that purge retains preserved forest analyses and evicts other results."""
    first = _completed(
        analysis(on=Forest)(lambda connector, forest: {'result': 'first'})
    )
    second = _completed(
        analysis(on=Forest)(lambda connector, forest: {'result': 'second'})
    )
    cache = _cache()
    cache.set(first)
    cache.set(second)
    cache.tracker.increment_forest_rev()
    cache.preserve(
        None, [first.passe.proto], lambda _: first.passe.proto
    )

    cache.purge()

    assert cache.get((first.passe, first.pointer, {})) is first
    assert cache.get((second.passe, second.pointer, {})) is None
    assert len(cache) == 1


def test_cache_purge_keeps_forest_results_preserved_across_each_revision():
    """Verify that preservation at every transition keeps an older forest result."""
    first = _completed(
        analysis(on=Forest)(lambda connector, forest: {'result': 'first'})
    )
    second = _completed(
        analysis(on=Forest)(lambda connector, forest: {'result': 'second'})
    )
    cache = _cache()
    cache.set(first)
    cache.tracker.increment_forest_rev()
    cache.preserve(
        None, [first.passe.proto], lambda _: first.passe.proto
    )
    cache.set(second)
    cache.tracker.increment_forest_rev()
    cache.preserve(
        None, [first.passe.proto], lambda _: first.passe.proto
    )

    cache.purge()

    assert cache.get((first.passe, first.pointer, {})) is first
    assert cache.get((second.passe, second.pointer, {})) is None
    assert len(cache) == 1


def test_cache_purge_keeps_tree_results_only_in_preserved_scopes():
    """Verify that purge validates tree analyses against their own tree revisions."""
    pm = PassManager()
    first_tree = Tree(ast.parse('x = 1'), 'first')
    second_tree = Tree(ast.parse('x = 2'), 'second')
    pm.add(first_tree)
    pm.add(second_tree)

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        return {'result': tree.identifier}

    first = pm.run(inspect_tree, first_tree)
    second = pm.run(inspect_tree, second_tree)
    cache = _cache()
    cache.set(first)
    cache.set(second)
    cache.tracker.increment_tree_rev(first_tree)
    cache.tracker.increment_tree_rev(second_tree)
    cache.preserve(
        first_tree, [inspect_tree], lambda _: inspect_tree
    )

    cache.purge()

    assert cache.get((first.passe, first.pointer, {})) is first
    assert cache.get((second.passe, second.pointer, {})) is None
    assert len(cache) == 1


def test_cache_purge_requires_preservation_for_every_tree_dependency():
    """Verify that a dependent tree result is evicted when one scope is not preserved."""
    pm = PassManager()
    first_tree = Tree(ast.parse('x = 1'), 'first')
    second_tree = Tree(ast.parse('x = 2'), 'second')
    pm.add(first_tree)
    pm.add(second_tree)

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        return {'result': tree.identifier}

    @analysis(on=Tree)
    def inspect_with_dependency(connector, tree):
        if tree is first_tree:
            connector.gather(inspect_tree, second_tree)
        return {'result': tree.identifier}

    local_result = pm.run(inspect_tree, first_tree)
    dependent_result = pm.run(inspect_with_dependency, first_tree)
    cache = _cache()
    cache.set(local_result)
    cache.set(dependent_result)
    dependency_snapshot = cache._revision_snapshot(dependent_result)
    assert {tree for tree, _ in dependency_snapshot[1]} == {
        first_tree,
        second_tree,
    }

    cache.tracker.increment_tree_rev(first_tree)
    cache.tracker.increment_tree_rev(second_tree)
    cache.preserve(
        first_tree,
        [inspect_tree, inspect_with_dependency],
        lambda _: inspect_tree,
    )

    cache.purge()

    assert cache.get((local_result.passe, local_result.pointer, {})) is local_result
    assert (
        cache.get(
            (dependent_result.passe, dependent_result.pointer, {})
        )
        is None
    )
    assert len(cache) == 1


def test_cache_purge_checks_preservation_history_for_forest_results():
    """Verify that preservation-history cleanup accounts for cached forest results."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.results_preservation[None] = {
        0: PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe])
    }

    cache.purge()

    assert None in cache.results_preservation
    assert cache.results_preservation[None] == {}


def test_cache_purge_compacts_tree_preservation_history():
    """Verify that explicit purge prunes local histories after cached results are checked."""
    completed = _completed(target='tree')
    tree = completed.pointer[1]
    cache = _cache()
    cache.set(completed)
    cache.results_preservation[tree] = {
        0: PreservedAnalyses(lambda name: completed.passe.proto, [completed.passe])
    }

    cache.purge()

    assert cache.results_preservation[tree] == {}


def test_cache_clear_removes_results_revisions_and_preservations():
    """Verify that clearing the cache empties every cache-owned store."""
    completed = _completed()
    cache = _cache()
    cache.set(completed)
    cache.results_preservation[None] = {}

    cache.clear()

    assert cache.results_store == {}
    assert cache.results_revision == {}
    assert cache.results_preservation == {}


def test_cache_records_preservation_history():
    """Verify that preservation declarations can be added to a revision history."""
    cache = _cache()
    passe = _analysis_pass()
    cache.tracker.increment_forest_rev()
    cache.tracker.increment_forest_rev()
    cache.preserve(None, [passe], lambda _: passe)

    assert passe() in cache.results_preservation[None][2]


def test_cache_length_reports_stored_results():
    """Verify that the cache length reflects its stored result count."""
    cache = _cache()
    assert len(cache) == 0

    cache.set(_completed())

    assert len(cache) == 1


def test_revision_tracker_starts_at_zero_and_increments_forest_revision():
    """Verify that forest revisions begin at zero and increment independently."""
    tracker = RevisionTracker()

    assert tracker.forest_rev() == 0
    tracker.increment_forest_rev()
    assert tracker.forest_rev() == 1


def test_revision_tracker_increments_forest_and_tree_revisions():
    """Verify that changing a tree increments both its local and the forest revision."""
    tracker = RevisionTracker()
    tree = Tree(ast.parse('x = 1'), 'mod')

    assert tracker.forest_rev() == 0
    assert tracker.tree_rev(tree) == 0
    tracker.increment_tree_rev(tree)
    assert tracker.forest_rev() == 1
    assert tracker.tree_rev(tree) == 1


def test_revision_tracker_tracks_distinct_trees_independently():
    """Verify that a tree edit does not increment another tree's local revision."""
    tracker = RevisionTracker()
    first = Tree(ast.parse('x = 1'), 'first')
    second = Tree(ast.parse('y = 2'), 'second')

    tracker.increment_tree_rev(first)

    assert tracker.tree_rev(first) == 1
    assert tracker.tree_rev(second) == 0
    assert tracker.forest_rev() == 1


def test_revision_tracker_accepts_weak_references_to_trees():
    """Verify that trees can be keys in the tracker's weak revision map."""
    import weakref

    tree = Tree(ast.parse('x = 1'), 'mod')

    assert weakref.ref(tree)() is tree


def test_revision_tracker_can_increment_only_the_forest():
    """Verify that structural forest edits can advance without changing local tree revisions."""
    tracker = RevisionTracker()
    tree = Tree(ast.parse('x = 1'), 'mod')

    tracker.increment_forest_rev()

    assert tracker.forest_rev() == 1
    assert tracker.tree_rev(tree) == 0



def test_revision_tracker_drops_collected_tree_revisions():
    """Verify that weakly tracked trees do not remain in the tracker after collection."""
    import gc
    import weakref

    tracker = RevisionTracker()
    tree = Tree(ast.parse('x = 1'), 'mod')
    tracker.increment_tree_rev(tree)
    reference = weakref.ref(tree)
    del tree
    gc.collect()

    assert reference() is None
    assert len(tracker._revisions) == 0


def test_cache_plugin_reuses_analysis_results():
    """Verify that repeated identical analysis requests return the cached result."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        calls.append(None)
        return {'result': len(forest)}

    pm = PassManager([PassManagerCache])

    completed = pm.run(inspect_forest)

    assert pm.cache.get((completed.passe, completed.pointer, {})) is completed
    assert pm.gather(inspect_forest) == 0
    assert pm.gather(inspect_forest) == 0
    assert calls == [None]


def test_cache_plugin_invalidates_forest_analysis_after_tree_edit():
    """Verify that a local tree transformation invalidates forest-wide analyses."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        calls.append(None)
        return {'result': len(calls)}

    @transformation(on=Tree)
    def change_tree(connector, tree):
        return {'update': True}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    assert pm.gather(inspect_forest) == 1
    pm.apply(change_tree, tree)

    assert pm.gather(inspect_forest) == 2
    assert calls == [None, None]


def test_cache_plugin_keeps_analysis_after_noop_transformation():
    """Verify that a transformation reporting no update leaves cached analyses valid."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(None)
        return {'result': len(calls)}

    @transformation(on=Tree)
    def noop(connector, tree):
        return {'update': False}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    assert pm.gather(inspect_tree, tree) == 1
    pm.apply(noop, tree)

    assert pm.gather(inspect_tree, tree) == 1
    assert calls == [None]


def test_cache_plugin_invalidates_forest_analysis_after_tree_removal():
    """Verify that removing a tree invalidates forest-wide analyses."""
    calls = []

    @analysis(on=Forest)
    def count_trees(connector, forest):
        calls.append(None)
        return {'result': len(forest)}

    pm = PassManager([PassManagerCache])
    first = Tree(ast.parse('x = 1'), 'first')
    second = Tree(ast.parse('y = 2'), 'second')
    pm.add(first)
    pm.add(second)

    assert pm.gather(count_trees) == 2
    pm.remove(second)

    assert pm.gather(count_trees) == 1
    assert calls == [None, None]


def test_cache_plugin_raises_on_cache_only_miss_without_running_analysis():
    """Verify that cache-only requests fail without executing an uncached analysis."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        calls.append(None)
        return {'result': 1}

    pm = PassManager([PassManagerCache])

    with pytest.raises(CacheMissError, match='no valid cached result'):
        pm.gather(inspect_forest, cache_only=True)

    assert calls == []


def test_cache_plugin_serves_cache_only_hit():
    """Verify that a cache-only request returns an existing analysis result."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        calls.append(None)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])

    assert pm.gather(inspect_forest) == 1
    assert pm.gather(inspect_forest, cache_only=True) == 1
    assert calls == [None]


def test_cache_plugin_does_not_cache_transformations():
    """Verify that transformations always execute, including no-op transformations."""
    calls = []

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        calls.append(None)
        return {'update': False}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    assert pm.apply(transform_tree, tree) is False
    assert pm.apply(transform_tree, tree) is False

    assert calls == [None, None]


def test_cache_plugin_rejects_cache_only_transformations():
    """Verify that cache-only mode cannot silently run a transformation."""

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        return {'update': False}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    with pytest.raises(CacheMissError, match='transformations are not cached'):
        pm.apply(transform_tree, tree, cache_only=True)


@pytest.mark.parametrize('preserve', [5, 'analysis', b'analysis'])
def test_cache_plugin_rejects_non_iterable_preservation_declarations(preserve):
    """Verify that transformations must declare preserved analyses as an iterable."""

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        return {'update': True, 'preserve': preserve}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    with pytest.raises(TypeError, match="transformation 'preserve' must be an iterable"):
        pm.apply(transform_tree, tree)


def test_cache_plugin_reports_unknown_preserved_aliases():
    """Verify that invalid aliases in preservation declarations raise a clear error."""

    @transformation(on=Tree)
    def transform_tree(connector, tree):
        return {'update': True, 'preserve': ('unknown-analysis',)}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    with pytest.raises(TypeError, match="invalid transformation 'preserve' declaration"):
        pm.apply(transform_tree, tree)


def test_cache_plugin_separates_results_by_run_options():
    """Verify that distinct run options do not share an analysis result."""
    calls = []

    @analysis(on=Forest)
    def inspect_forest(connector, forest):
        calls.append(None)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])

    assert pm.gather(inspect_forest, selector=frozenset({1})) == 1
    assert pm.gather(inspect_forest, selector=frozenset({1})) == 1
    assert pm.gather(inspect_forest, selector=frozenset({2})) == 2
    assert calls == [None, None]


def test_cache_plugin_does_not_share_tree_analysis_results_between_trees():
    """Verify that identical tree analyses retain separate results for different trees."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(tree.identifier)
        return {'result': tree.identifier}

    pm = PassManager([PassManagerCache])
    first = Tree(ast.parse('x = 1'), 'first')
    second = Tree(ast.parse('x = 2'), 'second')
    pm.add(first)
    pm.add(second)

    assert pm.gather(inspect_tree, first) == 'first'
    assert pm.gather(inspect_tree, second) == 'second'
    assert pm.gather(inspect_tree, first) == 'first'
    assert calls == ['first', 'second']


def test_cache_plugin_invalidates_analysis_after_tree_transformation():
    """Verify that a local result is evicted lazily after its tree changes."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(None)
        return {'result': len(calls)}

    @transformation(on=Tree)
    def change_tree(connector, tree):
        return {'update': True}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    assert pm.gather(inspect_tree, tree) == 1
    assert pm.apply(change_tree, tree) is True
    assert len(pm.cache) == 1
    assert pm.gather(inspect_tree, tree) == 2
    assert calls == [None, None]


def test_cache_plugin_purges_stale_results_only_when_requested():
    """Verify that explicit purge removes stale results without running analyses."""
    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        return {'result': tree.identifier}

    @transformation(on=Tree)
    def change_tree(connector, tree):
        return {'update': True}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)
    pm.gather(inspect_tree, tree)
    pm.apply(change_tree, tree)

    assert len(pm.cache) == 1

    pm.cache.purge()

    assert len(pm.cache) == 0


def test_cache_plugin_preserves_declared_tree_analyses():
    """Verify that a transformation can preserve a tree analysis across a tree revision."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(None)
        return {'result': len(calls)}

    @transformation(on=Tree)
    def change_tree(connector, tree):
        return {'update': True, 'preserve': (inspect_tree,)}

    pm = PassManager([PassManagerCache])
    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    assert pm.gather(inspect_tree, tree) == 1
    assert pm.apply(change_tree, tree) is True
    assert pm.gather(inspect_tree, tree) == 1
    assert pm.apply(change_tree, tree) is True
    assert pm.apply(change_tree, tree) is True
    assert pm.apply(change_tree, tree) is True
    assert pm.gather(inspect_tree, tree) == 1
    assert calls == [None]


def test_cache_plugin_keeps_unrelated_tree_analysis_after_local_edit():
    """Verify that editing one tree leaves another tree's local analysis cached."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(tree.identifier)
        return {'result': tree.identifier}

    @transformation(on=Tree)
    def change_tree(connector, tree):
        return {'update': True}

    pm = PassManager([PassManagerCache])
    first = Tree(ast.parse('x = 1'), 'first')
    second = Tree(ast.parse('x = 2'), 'second')
    pm.add(first)
    pm.add(second)

    assert pm.gather(inspect_tree, first) == 'first'
    assert pm.gather(inspect_tree, second) == 'second'
    pm.apply(change_tree, first)

    assert pm.gather(inspect_tree, second) == 'second'
    assert calls == ['first', 'second']


@pytest.mark.parametrize(
    'pass_options',
    [{}, {'_builtin_forest_transformation': False}, {'_builtin_forest_transformation': 1}],
)
def test_cache_plugin_rejects_unmarked_forest_transformations_before_execution(
    pass_options,
):
    """Verify that unsupported forest transformations fail before running."""
    calls = []

    @transformation(on=Forest, **pass_options)
    def change_forest(connector, forest):
        calls.append(None)
        return {'update': False}

    pm = PassManager([PassManagerCache])

    with pytest.raises(
        TypeError,
        match='forest-wide transformations are unsupported',
    ):
        pm.apply(change_forest)

    with pytest.raises(
        TypeError,
        match='forest-wide transformations are unsupported',
    ):
        pm.apply(change_forest, cache_only=True)

    assert calls == []


def test_cache_plugin_preserves_and_parameter_patterns():
    """Verify that transformation 'preserve' match their configured analysis instances."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree, *, optional=0):
        calls.append(optional)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])

    @transformation(on=Tree)
    def preserve_even(connector, tree):
        return {
            'update': True,
            'preserve': (
                PassPattern.like(
                    inspect_tree,
                    optional=lambda value: value % 2 == 0,
                ),
            ),
        }

    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    configured_instance = inspect_tree(optional=2)
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_even, tree) is True
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_even, tree) is True
    assert pm.gather(inspect_tree(optional=3), tree) == 2


def test_cache_plugin_preserves_aliases_and_parameter_patterns():
    """Verify that transformation 'preserve' match their configured analysis instances through string aliases."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree, *, optional=0):
        calls.append(optional)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])
    pm.configure(inspect_tree, 'inspect')
    configured_inspect = pm.get_passe('inspect')

    @transformation(on=Tree)
    def preserve_even(connector, tree):
        return {
            'update': True,
            'preserve': (
                PassPattern.like(
                    'inspect',
                    optional=lambda value: value % 2 == 0,
                ),
            ),
        }

    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    configured_instance = configured_inspect(optional=2)
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_even, tree) is True
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_even, tree) is True
    assert pm.gather(configured_inspect(optional=3), tree) == 2

def test_cache_plugin_preserves():
    """Verify that transformation 'preserve' analysis instances."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree, *, optional=0):
        calls.append(optional)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])

    @transformation(on=Tree)
    def preserve_inspect(connector, tree):
        return {
            'update': True,
            'preserve': (inspect_tree,)
        }

    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    configured_instance = inspect_tree(optional=2)
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_inspect, tree) is True
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_inspect, tree) is True
    assert pm.gather(inspect_tree(optional=3), tree) == 2

def test_cache_plugin_preserves_aliases():
    """Verify that transformation 'preserve' analysis instances through string aliases."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree, *, optional=0):
        calls.append(optional)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])
    pm.configure(inspect_tree, 'inspect')
    configured_inspect = pm.get_passe('inspect')

    @transformation(on=Tree)
    def preserve_inspect(connector, tree):
        return {
            'update': True,
            'preserve': ('inspect',)
        }

    tree = Tree(ast.parse('x = 1'), 'mod')
    pm.add(tree)

    configured_instance = configured_inspect(optional=2)
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_inspect, tree) is True
    assert pm.gather(configured_instance, tree) == 1
    assert pm.apply(preserve_inspect, tree) is True
    assert pm.gather(configured_inspect(optional=3), tree) == 2

def test_cache_plugin_keeps_tree_results_after_adding_tree(monkeypatch):
    """Verify that adding a tree preserves existing tree analyses without tree-wide work."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(None)
        return {'result': len(calls)}

    pm = PassManager([PassManagerCache])
    first = Tree(ast.parse('x = 1'), 'first')
    pm.add(first)
    tree_revision_calls = []
    monkeypatch.setattr(
        pm.cache.tracker,
        'increment_tree_rev',
        lambda tree: tree_revision_calls.append(tree),
    )

    assert pm.gather(inspect_tree, first) == 1
    pm.add(Tree(ast.parse('y = 2'), 'second'))

    assert pm.gather(inspect_tree, first) == 1
    assert calls == [None]
    assert tree_revision_calls == []


def test_cache_plugin_keeps_tree_results_after_removing_another_tree():
    """Verify that removing a tree preserves analyses for unchanged remaining trees."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(tree.identifier)
        return {'result': tree.identifier}

    pm = PassManager([PassManagerCache])
    first = Tree(ast.parse('x = 1'), 'first')
    second = Tree(ast.parse('y = 2'), 'second')
    pm.add(first)
    pm.add(second)

    assert pm.gather(inspect_tree, first) == 'first'
    pm.remove(second)

    assert pm.gather(inspect_tree, first) == 'first'
    assert calls == ['first']


@pytest.mark.parametrize('change', ['add', 'remove'])
def test_cache_plugin_keeps_complete_multi_tree_result_across_unrelated_membership_change(
    change,
):
    """Verify that independent tree paths survive unrelated forest membership changes."""
    calls = []

    @analysis(on=Tree)
    def inspect_dependency(connector, tree):
        calls.append(('dependency', tree.identifier))
        return {'result': tree.identifier}

    @analysis(on=Tree)
    def inspect_target(connector, tree):
        calls.append(('target', tree.identifier))
        connector.gather(inspect_dependency, 'dependency')
        return {'result': tree.identifier}

    pm = PassManager([PassManagerCache])
    target = Tree(ast.parse('x = 1'), 'target')
    dependency = Tree(ast.parse('y = 2'), 'dependency')
    unrelated = Tree(ast.parse('z = 3'), 'unrelated')
    pm.add(target)
    pm.add(dependency)
    if change == 'remove':
        pm.add(unrelated)

    completed = pm.run(inspect_target, target)
    assert completed.attributes['complete'] is True
    snapshot = pm.cache._revision_snapshot(completed)
    assert snapshot[0] is None
    assert {tree for tree, _ in snapshot[1]} == {target, dependency}

    if change == 'add':
        pm.add(unrelated)
    else:
        pm.remove(unrelated)

    assert pm.run(inspect_target, target) is completed
    assert calls == [('target', 'target'), ('dependency', 'dependency')]


def test_cache_plugin_invalidates_forest_analysis_after_adding_tree():
    """Verify that adding a tree invalidates cached forest-wide analyses."""
    calls = []

    @analysis(on=Forest)
    def count_trees(connector, forest):
        calls.append(None)
        return {'result': len(forest)}

    pm = PassManager([PassManagerCache])
    pm.add(Tree(ast.parse('x = 1'), 'first'))

    assert pm.gather(count_trees) == 1
    pm.add(Tree(ast.parse('y = 2'), 'second'))

    assert pm.gather(count_trees) == 2
    assert calls == [None, None]


def test_cache_plugin_invalidates_incomplete_forest_analysis_after_adding_tree():
    """Verify that adding a tree invalidates an incomplete forest analysis."""
    calls = []

    @analysis(on=Forest)
    def count_trees(connector, forest):
        calls.append(None)
        return {'result': len(forest), 'complete': False}

    pm = PassManager([PassManagerCache])
    pm.add(Tree(ast.parse('x = 1'), 'first'))

    assert pm.gather(count_trees) == 1
    pm.add(Tree(ast.parse('y = 2'), 'second'))

    assert pm.gather(count_trees) == 2
    assert pm.gather(count_trees) == 2
    assert calls == [None, None]


def test_cache_plugin_invalidates_incomplete_tree_analysis_after_adding_tree():
    """Verify that adding a tree invalidates an incomplete tree analysis."""
    calls = []

    @analysis(on=Tree)
    def inspect_tree(connector, tree):
        calls.append(tree.identifier)
        return {'result': len(calls), 'complete': False}

    pm = PassManager([PassManagerCache])
    target = Tree(ast.parse('x = 1'), 'target')
    pm.add(target)

    assert pm.gather(inspect_tree, target) == 1
    pm.add(Tree(ast.parse('y = 2'), 'new'))

    assert pm.gather(inspect_tree, target) == 2
    assert pm.gather(inspect_tree, target) == 2
    assert calls == ['target', 'target']


def test_cache_plugin_invalidates_incomplete_node_analysis_after_adding_tree():
    """Verify that adding a tree invalidates an incomplete node analysis."""
    calls = []

    @analysis(on=ast.Module)
    def inspect_module(connector, module):
        calls.append(None)
        return {'result': len(calls), 'complete': False}

    pm = PassManager([PassManagerCache])
    target = Tree(ast.parse('x = 1'), 'target')
    pm.add(target)

    assert pm.gather(inspect_module, target, target.root) == 1
    pm.add(Tree(ast.parse('y = 2'), 'new'))

    assert pm.gather(inspect_module, target, target.root) == 2
    assert pm.gather(inspect_module, target, target.root) == 2
    assert calls == [None, None]


def test_cache_plugin_does_not_cache_failed_analyses():
    """Verify that analysis failures are not inserted into the result cache."""
    calls = []

    @analysis(on=Forest)
    def fail_once(connector, forest):
        calls.append(None)
        raise RuntimeError('failure')

    pm = PassManager([PassManagerCache])
    for _ in range(2):
        with pytest.raises(RuntimeError, match='failure'):
            pm.gather(fail_once)

    assert calls == [None, None]