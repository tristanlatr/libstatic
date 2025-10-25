"""
Declares the forest and tree models as well as a couple of prootocols. 
"""
from __future__ import annotations

from enum import IntEnum

from typing import (
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Mapping,
    TypeVar,
    overload,
)

from libstatic._lib.structures import (
    FrozenDict,
)


############ Typing related declarations

type Element = Any
"""
Represent any element of the system: forest, tree, or any nodes.

An element must be weak referenciable: a tuple, a string or any 
other primitive types are NOT elements.
"""
type RootNode = Any
"Represent the root node of the tree (i.e. ast.Module)"

type Node = Any
"Represent any node in a tree, including it's root node"

_T = TypeVar("_T")


class Tree:
    """
    Encapsulate a single parse tree.

    All trees are required to have an identifier.
    This should be the python module name.

    @note: This is a read-only datastructure. Don't try to mutate identifier,
        root or attributes.
    """

    __slots__ = "__root", "__identifier", "attributes"

    # TODO: It would be best to provide a lazy loader for the parse tree, since
    # we'd always have to traverse the whole directory structure (because trees  
    # can't be bested under other trees), we want to avoid
    # parsing ASTs of trees that won't even be used because we scan for 
    # the whole standard library for instance. 
    def __init__(self, root: RootNode, identifier: str, 
                 **attributes: Hashable) -> None:
        
        self.__root = root
        self.__identifier = identifier
        self.attributes: Mapping[str, Hashable] = FrozenDict(**attributes)
        """
        Optional hashable metadata regarding this tree. 
        """

    @property
    def root(self) -> RootNode:
        """
        The root node of the tree.

        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = Tree(astmod, 'builtins')
        >>> assert tree.root is astmod
        """
        return self.__root

    @property
    def identifier(self) -> str:
        """
        The identifier of the module.

        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = Tree(astmod, 'builtins')
        >>> assert tree.identifier == 'builtins'
        >>> assert tree == Tree(astmod, 'builtins')
        >>> assert tree != Tree(ast.parse('...'), 'builtins')
        """
        return self.__identifier
    
    def __str__(self):
        return f'Tree {self.__identifier!r}'

    def __repr__(self):
        return f'Tree({self.__root!r}, {self.__identifier!r}, **{self.attributes!r})'

    def __hash__(self) -> int:
        return hash((self.root, self.identifier, self.attributes))

    def __eq__(self, other: object) -> bool:
        if isinstance(self, Tree) and isinstance(other, Tree):
            return (
                self.root == other.root
                and self.identifier == other.identifier
                and self.attributes == other.attributes
            )
        return NotImplemented


class TreeNotFound(KeyError):
    """
    A subclass of KeyError that is used when a tree is not found in the forest.
    """


class Forest(Collection[Tree]):
    """
    A collection of trees.

    Provides a mapping-ish interface to access the trees.
    You should not initiate a forest yourself, the passmanager will do it.

    Values can be accessed both by module name or by module ast node.

    Mutation methods (add/remove) are private since these action should only
    be performed through the passmanager L{add}/L{remove} methods or by 
    the approriate forest-wide transformations: L{add_tree} or L{remove_tree}.

    >>> import ast
    >>> astmod = ast.parse('x = 1')
    >>> tree = Tree(astmod, 'mod1', filename='./mod1.py')
    >>> pm = PassManager([tree])
    >>> isinstance(pm.trees, Forest)
    True
    >>> 'mod1' in pm.trees
    True
    >>> astmod in pm.trees
    True
    >>> tree in pm.trees
    True

    """

    __slots__ = "__identifier2tree", "__root2tree", "__trees"

    def __init__(self, trees: Iterable[Tree] | None = None) -> None:

        # each operation must maintain these 3 structures.
        self.__identifier2tree: dict[str, Tree] = {}
        self.__root2tree: dict[RootNode, Tree] = {}
        self.__trees: set[Tree] = set()

        if trees is not None:
            for t in trees:
                self._add(t)

    def _add(self, tree: Tree) -> None:
        # no-op is the tree is already in the collection.

        if tree in self.__trees:
            return

        if tree.identifier in self.__identifier2tree:
            raise ValueError(
                f"identifier {tree.identifier!r} "
                f"is already taken: {self[tree.identifier]}"
            )

        if tree.root in self.__root2tree:
            raise ValueError(
                f"root node {tree.identifier!r} is already "
                f"associated with another tree: {self[tree.root]}"
            )

        # add the tree in the collection.
        self.__identifier2tree[tree.identifier] = tree
        self.__root2tree[tree.root] = tree
        self.__trees.add(tree)

    def _remove(self, tree: Tree) -> None:
        if tree not in self:
            raise ValueError(f"tree not in the collection: {tree}")

        # remove the tree from the collection
        del self.__identifier2tree[tree.identifier]
        del self.__root2tree[tree.root]
        self.__trees.discard(tree)

    #  getitem interface

    def __getitem__(self, __key: str | RootNode) -> Tree:
        try:
            if isinstance(__key, str):
                return self.__identifier2tree[__key]
            else:
                return self.__root2tree[__key]
        except KeyError as e:
            raise TreeNotFound(__key) from e

    @overload
    def get(self, key: str | RootNode) -> Tree | None: ...
    @overload
    def get(self, key: str | RootNode, default: _T) -> Tree | _T: ...
    def get(self, key: str | RootNode, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # collection interface

    def __iter__(self) -> Iterator[Tree]:
        # we're not using __trees here because set is not ordered.
        # but dict is ordered. 
        return iter(self.__identifier2tree.values())

    def __len__(self) -> int:
        return len(self.__trees)

    def __contains__(self, other: object) -> bool:
        # A forest contains trees, root nodes and identifiers.
        return (
            other in self.__trees
            or other in self.__identifier2tree
            or other in self.__root2tree
        )

class Level(IntEnum):
    FOREST = 3
    TREE = 2
    NODE = 1

class PassKind(IntEnum):
    TRANSFORMATION = 10
    ANALYSIS = 11