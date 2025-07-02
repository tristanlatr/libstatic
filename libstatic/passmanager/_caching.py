"""
Builtin caching plugin for the standard AST module. 
Provides a simple and advanced stategies.
"""
from __future__ import annotations

import abc
from collections import deque
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from inspect import signature, Parameter
from itertools import chain, product
import weakref

from typing import (
    Callable,
    Collection,
    Container,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from typing import NotRequired, TypeAlias, TypedDict
else:
    final = lambda f: f
    TypedDict = object

from libstatic._lib.structures import (
    Cache,
    Indexer,
    FrozenDict,
    OrderedSet,
)

import attrs


type _CacheKeyT = tuple[
    PassInstance, int, bool, "Hashable | None", "Hashable | None"
]

class PassPattern:
    """
    Represents several derivations of the same pass prototype,
    with different parameters.
    Use the L{like()} method to create instances of this class.

    >>> import ast
    >>> @analysis(on=ast.AST)
    ... def count_objs(_, node: ast.AST, *, type:type=object):
    ...     yield 'result', len(set(n for n in ast.walk(node) if isinstance(n, type)))

    Let's declare a few testing patterns:

    >>> # this pattern will never match any pass instance
    >>> like_none = count_objs.like(type=(lambda _: False)) 
    >>> # this pattern will match any instances of the pass 'count_objs'
    >>> like_any = count_objs.like(type=(lambda _: True))  
    >>> # this one idem 
    >>> like_any2 = count_objs.like()                       
    >>> # this one will match only if type=ast.ClassDef.
    >>> like_cls = count_objs.like(type=(lambda v: v is ast.ClassDef))

    The pass pattern implement C{__eq__} wich makes it suitable for
    checking whether a pass instance is C{in} a container with a pattern in it.

    >>> count_objs() in [like_cls]
    False
    >>> count_objs(type=ast.AST) in [like_none]
    False
    >>> count_objs(type=ast.AST) in [like_any]
    True
    >>> count_objs(type=ast.AST) in [like_any2]
    True
    >>> count_objs(type=ast.AST) in [like_cls]
    False
    >>> count_objs(type=ast.ClassDef) in [like_cls]
    True

    A pass prototype never macthes against a pattern,
    only actual pass instances matches.

    >>> count_objs in [like_none, like_any2, like_any, like_cls]
    False

    The matching is ignored for missing arguments.

    >>> @analysis(on=ast.AST)
    ... def has_required_param(_, node, *, required):
    ...     yield 'result', 1
    >>> param_like_none = has_required_param.like(required=(lambda _: False))
    >>> has_required_param() in [param_like_none]
    True
    >>> has_required_param('anything') in [param_like_none]
    False

    The pattern can be generic by ommiting the argument.

    >>> has_required_param('anything') in [has_required_param.like()]
    True
    """
    __slots__ = "_match", "_passe"

    def __init__(
        self, passe: PassPrototype, **predicate: Callable[[object], bool]
    ) -> None:
        # TODO: Not all parameters might be given,
        # but we should still validate the name of the predicates!!!
        self._match = predicate
        self._passe = passe

    def matches(self, other: object) -> bool:
        """
        Whether the given pass instance matches the pattern.
        """
        if not isinstance(other, PassInstance):
            return False
        proto = other.proto
        # Two passes matches if they share the same prototype
        if proto != self._passe:
            return False

        args = {**proto.optional_params, **other.args}
        # And all the arguments predicates returns a truthy value,
        # for argument that are set, for the one that are eventually missing from
        # the given PassInstance, the matching is ignored.
        for k, cb in self._match.items():
            if (k in args) and (not cb(args[k])):
                return False
        return True

    __hash__ = None # This class is not hashable by nature.
    __eq__ = matches

class PreservedAnalyses:
    """
    Container for checking whether a given analyse is preserved after a
    transformatation has been applied. 
    """

    def __init__(self, get_passe: Callable[[str], PassLike], 
                 analyses: Iterable[PassPattern | PassLike | str] = ()):
        self.__passes: set[PassInstance] = set()

        # In order to to avoid linear time complexity based on the whole list
        # of patterns, we index the patterns based on the "pass prototype"
        # in a dict and matche only the subset coresponding.
        self.__patterns: dict[PassPrototype, list[PassPattern]] = {}
        self.__get_passe = get_passe
        for a in analyses:
            self.add(a)
    
    def add(self, pass_or_pattern: PassLike | PassPattern | str) -> None:
        if isinstance(pass_or_pattern, PassPrototype):
            self.__passes.add(pass_or_pattern())
        elif isinstance(pass_or_pattern, PassInstance):
            self.__passes.add(pass_or_pattern)
        elif isinstance(pass_or_pattern, str):
            self.add(self.__get_passe(pass_or_pattern))
        else:
            proto = pass_or_pattern._passe
            self.__patterns.setdefault(proto, []).append(pass_or_pattern)

    # because the manner the pattern matching is implemented (with __eq__)
    # it's not possible to simply remove a pattern element from a list
    # given an existing equivalent pattern instance. 
    # I choosed not to implement the remove() operation because it adds complexity, 
    # for someting without any usage at the moment.
    
    def __contains__(self, other: PassInstance) -> bool:
        return (other in self.__passes) or (
                (p:=other.proto) in self.__patterns 
                and other in self.__patterns[p])

CACHE_KEYS = OrderedSet(
    (   "passe",  # PassInstance
        "knowledge",  # integer: FOREST / TREE / NODE
        "completeness",  # boolean
        "tree",  # Tree or None
        "node",  # AnyNode or None
    ))

# TODO: The revision tracker should follow the import graph so we can preserve more
# forest-knowledge analyses. We can even automate this by using the used_paths combined 
# with a regular import analysis in order to detect wether the analyses followed the
# imports or not, and a flag can be added to the cache. This manner we can significantly
# reduce the overhead introduced by the multiplication of forest-knowledge analyses
class RevTracker:
    """
    The revision tracker is a part of the advanced caching features.

    Tracks the revision of all elements: the forest itself, trees and nodes.
    It MUST be informed with method `increment_rev` when a transformation occurs.
    Then it increments the relevant revisions hierachically in order to keep unrelated 
    nodes' revision unchanged. 

    One instance of this class can only be used to track revisions of elements 
    in single forest, it's also assumed that the tree instances are not shared across 
    multiple forests and all transformation are duly signaled to the instance of this class.
    """
    
    def __init__(self, get_anscestors: Callable[[Tree, Node], Iterable[Node]]):
        """
        :param get_anscestors: A callable that, given a tree and a 
            node that lives inside it, returns the ancestors nodes
            as a sequence from the root to the node's direct parent. 
        """
        self._forest_sentinel = object() # the forest object version
        self._get_anscestors = get_anscestors # dependency injection

        # We abuse the complex type as a two dimentional vector of ints.
        self._revisions: dict[Element, complex] = weakref.WeakKeyDictionary()
    
    def _elements(self, 
                  tree: Tree | None = None, 
                  node: Node | None = None) -> deque[object]:
        """
        Elements in this order: 

            - Node
            - Node ancestors up to the Module
            - Tree the node lives in
            - Forest (sentinel). 

        The returned deque will always at least contain the fores instance. 
        """
        elements = deque()
        if node:
            assert tree is not None
            # if there are nodes in between tree and node, add them to the stack
            elements.extendleft(self._get_anscestors(tree, node))
            elements.appendleft(node)
            # elements now contains the 
            # affected node first, then all it's ancestors.
        
        if tree:
            elements.append(tree)
        elements.append(self._forest_sentinel)
        return elements

    def increment_rev(self, 
                    tree: Tree | None = None, 
                    node: Node | None = None) -> None:
        """
        Signals that the given tree/node has been transformed 
        and increment it's revision hierachically. 
        
        It's illegal to give a node without a tree. 
        When no tree or node are given it wil be 
        interpreted as a forest transformation.
        """
        rev = self._revisions
        elements = self._elements(tree, node)
        
        def _incr(key: object, value: complex) -> None:
            rev.setdefault(key, 0j)
            rev[key] += value

        # the verion incrementation happends bottom-up, first we increment REAL part
        # of the directly affected element, then we increment the IMAGINARY part
        # of indirectly affected elements, up the global element version.
        _incr(elements.popleft(), 1)
        for e in elements:
            _incr(e, 1j)
    
    def rev(self, 
                 tree: Tree | None = None, 
                 node: Node | None = None, ) -> tuple[str, str, str]:
        """
        Computes the revision of the element at tree/.../node. 
        The revision is a three layered element.
        It's illegal to give a node without a tree. 
        When no tree or node are given the revision of the whole 
        forest will be returned.
        """
        rev = self._revisions
        elements = self._elements(tree, node)

        forest_rev = rev.get(elements.pop(), 0j)
        if not elements:
            return f"{forest_rev}", '0', '0'
        
        tree_rev = rev.get(elements.pop(), 0j)
        if not elements:
            return f"{forest_rev}", f"{tree_rev}", '0'
        
        # The version of a node is defined by the hash of it's own two dimentionned
        # vector (IMAGINARY and REAL) combined to the REAL dimention of the enclosing
        # elements. 
        node_rev = rev.get(elements.popleft(), 0j)
        parents_rev = (rev.get(e, 0j).real for e in elements)
        
        return f"{forest_rev}", f"{tree_rev}", '/'.join(map(str, chain(parents_rev, node_rev)))

@attrs.frozen(slots=True)
class PassManagerCache:
    """
    Wraps the generic L{Cache} class for caching L{CompletedPass} instances.
    """

    _cache: Cache[_CacheKeyT, CompletedPass]
    tracker: RevTracker

    def get(self, passe: PassInstance, pointer: _ElementPath) -> CompletedPass | None:
        for k in self._mk_cache_keys_to_get_result(passe, pointer):
            if result := self._cache.get(k):
                # the result is cached :)
                return result
        return None

    def set(self, result: CompletedPass):
        key = self._mk_cache_key_to_set_result(result)
        self._cache.set(key, result)

    @staticmethod
    def _mk_cache_key_to_set_result(result: CompletedPass) -> _CacheKeyT:
        pointer: _ElementPath = result.pointer
        # we do not use the forest part of the pointer here
        path = pointer + (None,) * (3 - len(pointer))
        completeness = True
        return result.passe, result.knowledge, completeness, path[1], path[2]

    @staticmethod
    def _mk_cache_keys_to_get_result(
        passe: PassInstance,
        pointer: _ElementPath,
    ) -> Iterator[_CacheKeyT]:
        # we do not use the forest part of the pointer here
        path = pointer + (None,) * (3 - len(pointer))
        p1 = path[1]
        p2 = path[2]
        runs_on = int(passe.proto.runs_on)

        yield passe, runs_on, True, p1, p2,
        while runs_on < Level.FOREST:
            runs_on += 1
            yield passe, runs_on, True, p1, p2,
        # The completeness can only be False for forest knowledge analyses.
        yield passe, runs_on, False, p1, p2,

    # TODO: There is too much boilerplate code around the cache management...
    def remove(self, key: _CacheKeyT) -> None:
        self._cache.remove(key)
    remove.__doc__ = Cache.remove.__doc__
    
    def search(self, **key) -> Collection:
        return self._cache.search(**key)
    search.__doc__ = Cache.search.__doc__
    
    def allkeys(self) -> Collection: # used for testing
        return self._cache.allkeys()
    allkeys.__doc__ = Cache.allkeys.__doc__
