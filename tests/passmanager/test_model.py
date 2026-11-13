"""Tests for the passmanager model types."""

import ast

import pytest

from libstatic.passmanager._model import (
    Forest,
    Level,
    PassKind,
    Tree,
    TreeNotFound,
)


def _tree(name: str, source: str = 'x = 1', **attributes) -> Tree:
    return Tree(ast.parse(source), name, **attributes)


def test_tree_exposes_its_root():
    """Verify that a tree exposes the root node it was created with."""
    root = ast.parse('x = 1')
    tree = Tree(root, 'mod1')

    assert tree.root is root


def test_tree_exposes_its_identifier():
    """Verify that a tree exposes its module identifier."""
    tree = _tree('mod1')

    assert tree.identifier == 'mod1'


def test_tree_preserves_attributes():
    """Verify that a tree exposes its metadata attributes."""
    tree = _tree('mod1', filename='mod1.py', version=1)

    assert dict(tree.attributes) == {'filename': 'mod1.py', 'version': 1}


def test_tree_string_representation():
    """Verify that a tree's string form identifies its module."""
    tree = _tree('mod1')

    assert str(tree) == "Tree 'mod1'"


def test_tree_debug_representation():
    """Verify that a tree's debug representation includes its type and root."""
    tree = _tree('mod1')

    assert repr(tree).startswith('Tree(')
    assert repr(tree).endswith("**{})")


def test_trees_compare_by_root_identifier_and_attributes():
    """Verify that trees with matching fields compare equal and differing trees do not."""
    root = ast.parse('x = 1')
    tree = Tree(root, 'mod1', filename='mod1.py')
    same_tree = Tree(root, 'mod1', filename='mod1.py')
    different_tree = _tree('mod1', 'y = 2', filename='mod1.py')

    assert tree == same_tree
    assert tree != different_tree
    assert tree != object()


def test_tree_hash_uses_its_identity_fields():
    """Verify that a tree's hash is derived from its root, identifier, and attributes."""
    root = ast.parse('x = 1')
    tree = Tree(root, 'mod1', filename='mod1.py')

    assert hash(tree) == hash((root, 'mod1', tree.attributes))


def test_forest_starts_empty():
    """Verify that a newly created forest contains no trees."""
    assert len(Forest()) == 0


def test_forest_preserves_tree_insertion_order():
    """Verify that forest iteration follows the order in which trees were added."""
    tree1 = _tree('mod1')
    tree2 = _tree('mod2', 'y = 2')
    forest = Forest([tree1, tree2])

    assert list(forest) == [tree1, tree2]


def test_forest_length_tracks_added_trees():
    """Verify that adding distinct trees increases the forest size."""
    tree1 = _tree('mod1')
    tree2 = _tree('mod2', 'y = 2')
    forest = Forest([tree1, tree2])

    assert len(forest) == 2


def test_adding_existing_tree_is_a_noop():
    """Verify that adding the same tree twice does not duplicate it."""
    tree = _tree('mod1')
    forest = Forest([tree])

    forest._add(tree)

    assert len(forest) == 1


def test_forest_contains_tree_identifiers():
    """Verify that a forest recognizes stored tree identifiers."""
    tree = _tree('mod1')
    forest = Forest([tree])

    assert 'mod1' in forest


def test_forest_contains_tree_roots():
    """Verify that a forest recognizes stored tree root nodes."""
    tree = _tree('mod1')
    forest = Forest([tree])

    assert tree.root in forest


def test_forest_contains_trees():
    """Verify that a forest recognizes its stored tree objects."""
    tree = _tree('mod1')
    forest = Forest([tree])

    assert tree in forest


def test_forest_does_not_contain_unknown_identifiers():
    """Verify that a forest rejects unknown identifiers in membership checks."""
    assert 'missing' not in Forest()


def test_forest_looks_up_trees_by_identifier():
    """Verify that a tree can be retrieved from a forest by its identifier."""
    tree = _tree('mod1')
    forest = Forest([tree])

    assert forest['mod1'] is tree


def test_forest_looks_up_trees_by_root():
    """Verify that a tree can be retrieved from a forest by its root node."""
    tree = _tree('mod1')
    forest = Forest([tree])

    assert forest[tree.root] is tree


def test_forest_get_returns_none_for_unknown_identifier():
    """Verify that get returns None for an unknown identifier by default."""
    assert Forest().get('missing') is None


def test_forest_get_returns_a_supplied_default():
    """Verify that get returns the supplied default for an unknown identifier."""
    assert Forest().get('missing', 'fallback') == 'fallback'


def test_forest_lookup_raises_for_unknown_identifier():
    """Verify that indexing by an unknown identifier raises TreeNotFound."""
    with pytest.raises(TreeNotFound, match='missing'):
        Forest()['missing']


def test_forest_lookup_raises_for_unknown_root():
    """Verify that indexing by an unknown root node raises TreeNotFound."""
    with pytest.raises(TreeNotFound):
        Forest()[ast.parse('x = 1')]


def test_forest_removes_a_stored_tree():
    """Verify that removing a tree deletes its identifier from the forest."""
    tree = _tree('mod1')
    forest = Forest([tree])

    forest._remove(tree)

    assert 'mod1' not in forest


def test_forest_remove_rejects_an_unknown_tree():
    """Verify that removing a tree absent from the forest raises ValueError."""
    with pytest.raises(ValueError, match='tree not in the collection'):
        Forest()._remove(_tree('mod1'))


def test_forest_rejects_duplicate_identifiers():
    """Verify that a forest cannot contain distinct trees with the same identifier."""
    tree = _tree('mod1')
    forest = Forest([tree])

    with pytest.raises(ValueError, match="identifier 'mod1' is already taken"):
        forest._add(_tree('mod1', 'y = 2'))


def test_forest_rejects_duplicate_roots():
    """Verify that a root node cannot be associated with multiple trees."""
    tree = _tree('mod1')
    forest = Forest([tree])

    with pytest.raises(ValueError, match='root node'):
        forest._add(Tree(tree.root, 'mod2'))


def test_level_values():
    """Verify the declared ordering of pass execution levels."""
    assert list(Level) == [Level.FOREST, Level.TREE, Level.NODE]
    assert Level.FOREST > Level.TREE > Level.NODE


def test_pass_kind_values():
    """Verify the pass kinds."""
    assert list(PassKind) == [PassKind.TRANSFORMATION, PassKind.ANALYSIS]
