"""
Builtin caching plugin for the standard AST module. 
Provides a simple and advanced stategies.
"""
from __future__ import annotations

import weakref

from typing import (
    Callable,
    Iterable,
    Any,
    Hashable,
    Mapping,
    MutableMapping,
)

import attrs

from ._passe import PassInstance, PassPrototype, PassLike
from ._model import Level, Tree, Forest, Element, Node, PassKind

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
        known_parameters = set(passe.params) | set(passe.optional_params)
        unknown_parameters = predicate.keys() - known_parameters
        if unknown_parameters:
            # Fail at declaration time so a typo cannot silently preserve nothing.
            unknown = sorted(unknown_parameters)[0]
            raise ValueError(f"unknown pass parameter {unknown!r}")
        if any(not callable(callback) for callback in predicate.values()):
            raise TypeError("pass pattern predicates must be callable")
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

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        return self.matches(other)

    # Method to create a pattern from this pass.
    @classmethod
    def like(cls, passe: PassPrototype,  **predicate: Callable[[object], bool]) -> PassPattern:
        """
        Create a pattern representing several possible derivations of
        the pass to be matched against other passes.

        Designed to be used for preserved analyses.

        @param predicate: The analysis parameters names to the match function.
            A match function is a one-argument
            callable that returne whether the value for the parameter matches.
        """
        return PassPattern(passe, **predicate)

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
        elif isinstance(pass_or_pattern, PassPattern):
            proto = pass_or_pattern._passe
            self.__patterns.setdefault(proto, []).append(pass_or_pattern)
        else:
            raise TypeError(
                f"expected a pass prototype, pass instance, alias, or pattern; "
                f"got {type(pass_or_pattern).__name__}"
            )

    # because the manner the pattern matching is implemented (with __eq__)
    # it's not possible to simply remove a pattern element from a list
    # given an existing equivalent pattern instance. 
    # I choosed not to implement the remove() operation because it adds complexity, 
    # for someting without any usage at the moment.
    
    def __contains__(self, other: PassInstance) -> bool:
        return (other in self.__passes) or (
                (p:=other.proto) in self.__patterns 
                and other in self.__patterns[p])

# For a **phase one** caching strategy
# TODO: Make a SimpleRevTracker that do not need to understand the node hierarchy,
# simply incrementing the module revision anytime anything changes in that module. 

# Revision based caching should still allow some transforamtions to PRESERVE other
# results in cache. This is implemented by: for each result in cache, remembering the revision
# of the tree it has been run onto. When a transformation preserves results, we store that
# informations in another structure that says that result is still valid for that revision
# eventho it has been ran in a older revision. 
from ._passmanager import (
    CompletedPass,
    FailedPass,
    IPluginRegistrar,
    Trigger,
    UnpreparedPass,
    _HookedAll,
    _PassRun,
)


def _passrun_key(passerun: _PassRun) -> Hashable:
    passe, pointer, options = passerun
    key_options = dict(options)
    # This flag controls lookup behavior, not the analysis result itself.
    key_options.pop("cache_only", None)
    # frozenset keeps keyword order irrelevant and naturally rejects unhashable values.
    return passe, pointer, frozenset(key_options.items())


def _tree_for_pointer(pointer: tuple[Any, ...]) -> Tree | None:
    if len(pointer) > 1 and isinstance(pointer[1], Tree):
        return pointer[1]
    return None

@attrs.frozen(slots=True)
class PassManagerCache:
    """
    Caching L{CompletedPass} instances.
    """

    tracker: RevisonTracker
    results_store: dict[Hashable, CompletedPass]
    results_revision: dict[Hashable, RevisionSnapshot]
    results_preservation: dict[Tree | None, dict[int, PreservedAnalyses]]

    def purge(self) -> None:
        """
        Erase stale data and compact preservation histories.

        This intentionally performs a full cache scan and is therefore an
        explicit memory-management operation, not part of transformation hooks.
        """
        for key in tuple(self.results_store):
            self._get(key)
        self._purge_preservations()

    def clear(self) -> None:
        """
        Erase all data from the cache. 
        """
        self.results_preservation.clear()
        self.results_revision.clear()
        self.results_store.clear()

    def get(self, passerun: _PassRun) -> CompletedPass | None:
        """
        Request the cached result for this passerun. 

        If the cache contains the result but the result is stale, it will be cleared. 
        """
        return self._get(_passrun_key(passerun))

    def _evict(self, key: Hashable) -> None:
        self.results_store.pop(key, None)
        self.results_revision.pop(key, None)

    def _get(self, key: Hashable) -> CompletedPass | None:
        result = self.results_store.get(key)
        if result is None:
            return None

        cached_revision = self.results_revision.get(key)
        # Without a valid snapshot there is no evidence the result is current,
        # so discard it rather than risk returning stale analysis data.
        if cached_revision is None:
            self._evict(key)
            return None

        try:
            knowledge = Level(cached_revision[0])
        except ValueError:
            self._evict(key)
            return None

        passe_instance = result.passe
        scopes: tuple[tuple[Tree | None, int, int], ...]
        # Forest analyses use the forest epoch; tree and node analyses compare
        # every participating tree, including their target, at local epochs.
        if knowledge is Level.FOREST:
            current_forest_revision = self.tracker.rev(None)[0]
            start = cached_revision[1]
            end = current_forest_revision
            scopes = ((None, start, end),)
        else:
            scopes = tuple(
                (tree, revision, self.tracker.rev(tree)[1])
                for tree, revision in cached_revision[2]
            )

        for scope, start, end in scopes:
            if not self._preserved_across(
                passe_instance,
                self.results_preservation.get(scope, {}),
                start,
                end,
            ):
                self._evict(key)
                return None

        # Validating every intervening transition lets the entry advance to the
        # current snapshot and avoids replaying the same history next time.
        self.results_revision[key] = self._revision_snapshot(result)
        return result

    @staticmethod
    def _preserved_across(
        passe: PassInstance,
        history: Mapping[int, PreservedAnalyses],
        start: int,
        end: int,
    ) -> bool:
        if end < start:
            return False
        # Every transition matters: a later preservation cannot repair a result
        # invalidated by an earlier unpreserved change.
        return all(
            (preserved := history.get(revision)) is not None
            and passe in preserved
            for revision in range(start + 1, end + 1)
        )

    def _revision_snapshot(self, result: CompletedPass) -> RevisionSnapshot:
        # Dependencies can raise a pass's actual knowledge beyond its declared
        # scope, so snapshots follow the completed run's metadata.
        knowledge = Level(result.meta.knowledge)
        if knowledge is Level.FOREST:
            return knowledge, self.tracker.rev(None)[0], ()
        trees = {
            tree
            for path in result.meta.used_paths
            for tree in path[1:2]
            if isinstance(tree, Tree)
        }
        # Include the target tree even when it did not appear in used_paths.
        target_tree = _tree_for_pointer(result.pointer)
        if target_tree is not None:
            trees.add(target_tree)
        return knowledge, 0, tuple(
            sorted(
                ((tree, self.tracker.rev(tree)[1]) for tree in trees),
                key=lambda item: (item[0].identifier, id(item[0])),
            )
        )

    def _purge_preservations(self) -> None:
        # Keep transitions newer than the oldest live result snapshot; earlier
        # history cannot affect future validity checks for those results.
        forest_floors = [
            revision[1]
            for revision in self.results_revision.values()
            if revision[0] is Level.FOREST
        ]
        tree_floors: dict[Tree, int] = {}
        for knowledge, _, trees in self.results_revision.values():
            if knowledge is Level.FOREST:
                continue
            for tree, revision in trees:
                tree_floors[tree] = min(tree_floors.get(tree, revision), revision)

        for scope, history in tuple(self.results_preservation.items()):
            if scope is None:
                floor = min(forest_floors) if forest_floors else None
            else:
                floor = tree_floors.get(scope)
            if floor is None:
                del self.results_preservation[scope]
                continue
            self.results_preservation[scope] = {
                revision: preserved
                for revision, preserved in history.items()
                if revision > floor
            }

    def set(self, result: CompletedPass) -> None:
        """Store a completed pass result at the current target revision."""
        key = _passrun_key((result.passe, result.pointer, result.run_options))
        self.results_store[key] = result
        self.results_revision[key] = self._revision_snapshot(result)

    def preserve(
        self,
        tree: Tree | None,
        revision: int,
        analyses: Iterable[PassPattern | PassLike | str],
        get_passe: Callable[[str], PassLike],
    ) -> None:
        """Record analyses preserved at a forest or tree revision."""
        history = self.results_preservation.setdefault(tree, {})
        history[revision] = PreservedAnalyses(get_passe, analyses)

    def __len__(self) -> int:
        return len(self.results_store)


type Revision = tuple[int, int]
# Snapshot slot 1 is the forest epoch for forest analyses; slot 2 stores local
# tree epochs for tree/node analyses.
type RevisionSnapshot = tuple[Level, int, tuple[tuple[Tree, int], ...]]


class RevisonTracker:
    def __init__(self) -> None:
        self._forest_revision = 0
        self._revisions: MutableMapping[Tree, int] = weakref.WeakKeyDictionary()

    def increment_rev(self, tree: Tree | None) -> None:
        """Increment the forest revision and, when provided, the tree revision."""
        self._forest_revision += 1
        if tree is not None:
            self._revisions[tree] = self._revisions.get(tree, 0) + 1

    def increment_forest_rev(self) -> None:
        """Record a forest membership change without revising tree contents."""
        self._forest_revision += 1

    def increment_tree_rev(self, tree: Tree) -> None:
        """Increment only a tree's local revision."""
        self._revisions[tree] = self._revisions.get(tree, 0) + 1

    def rev(self, tree: Tree | None) -> Revision:
        """Return the forest and optional tree revisions."""
        forest_revision = self._forest_revision
        if tree is None:
            return forest_revision, 0
        return forest_revision, self._revisions.get(tree, 0)

RevisionTracker = RevisonTracker


@attrs.frozen(slots=True)
class PassManagerCaching:
    """Plugin that caches analysis results using forest and tree revisions."""

    name: str = "cache"
    cache: PassManagerCache = attrs.field(
        factory=lambda: PassManagerCache(RevisonTracker(), {}, {}, {})
    )
    _passmanager: Any = attrs.field(default=None, repr=False, eq=False)

    def register(self, registrar: IPluginRegistrar) -> Iterable[PassManagerCaching]:
        object.__setattr__(self, "_passmanager", registrar)
        registrar.hooks.install(
            self._before_pass,
            when=Trigger.UNPREPARED,
            kind=_HookedAll.ALL,
            level=_HookedAll.ALL,
        )
        registrar.hooks.install(
            self._after_pass,
            when=Trigger.COMPLETED,
            kind=_HookedAll.ALL,
            level=_HookedAll.ALL,
            knowledge=_HookedAll.ALL,
        )
        yield self

    def _before_pass(
        self, unprepared: UnpreparedPass
    ) -> CompletedPass | None:
        options = dict(unprepared.run_options)
        cache_only = bool(options.pop("cache_only", False))
        if (
            unprepared.passe.proto.kind is PassKind.TRANSFORMATION
            and unprepared.passe.proto.runs_on is Level.FOREST
            and unprepared.passe.proto.pass_options.get(
                "_cache_membership_only"
            )
            is not True
        ):
            # Validate before preparation so unsupported transforms cannot mutate
            # the forest, even when they would eventually report update=False.
            raise TypeError(
                "forest-wide transformations are unsupported unless "
                "_cache_membership_only=True"
            )
        if unprepared.passe.proto.kind.name != "ANALYSIS":
            # Transformations may change external state, so they always execute.
            if cache_only:
                raise CacheMissError(
                    "transformations are not cached and cannot run with cache_only=True"
                )
            return None
        key_options = options
        result = self.cache.get(
            (unprepared.passe, unprepared.pointer, key_options)
        )
        if result is None and cache_only:
            raise CacheMissError(
                f"no valid cached result for "
                f"{unprepared.passe!r} at {unprepared.pointer!r}"
            )
        return result

    def _after_pass(self, completed: CompletedPass) -> None:
        if completed.passe.proto.kind.name == "ANALYSIS":
            self.cache.set(completed)
            return

        if not completed.update:
            return

        level = completed.passe.proto.runs_on
        tree = _tree_for_pointer(completed.pointer)
        if level is Level.FOREST:
            # Only marked membership transforms reach this hook; they revise
            # the forest without walking or changing any individual tree.
            self.cache.tracker.increment_forest_rev()
        else:
            # A tree edit also advances the forest epoch because forest analyses
            # may depend on the edited tree.
            self.cache.tracker.increment_rev(tree)
        preserved = completed.attributes.get("preserved", ())
        if not isinstance(preserved, Iterable) or isinstance(preserved, (str, bytes)):
            raise TypeError("transformation 'preserved' must be an iterable")
        try:
            analyses = tuple(preserved)
            preserved_set = PreservedAnalyses(
                self._passmanager.get_passe, analyses
            )
        except (TypeError, ValueError, KeyError) as error:
            raise TypeError(f"invalid preserved analyses declaration: {error}") from error

        # Store history at each affected scope so lookups validate only the
        # forest or trees relevant to the cached analysis.
        forest_revision = self.cache.tracker.rev(None)[0]
        self.cache.results_preservation.setdefault(None, {})[forest_revision] = preserved_set
        if tree is not None:
            tree_revision = self.cache.tracker.rev(tree)[1]
            self.cache.results_preservation.setdefault(tree, {})[tree_revision] = preserved_set
        # Do not purge here: transformation hooks must stay O(1) in tree count.
        # Stale entries are evicted on lookup; callers explicitly purge to reclaim memory.


class CacheMissError(LookupError):
    """Raised when a cache-only pass request has no valid cached result."""

# For a **phase-two** caching strategy 
# TODO: The revision tracker should follow the import graph so we can preserve more
# forest-knowledge analyses. We can even automate this by using the used_paths combined 
# with a regular import analysis in order to detect wether the analyses followed the
# imports or not, and a flag can be added to the cache. This manner we can significantly
# reduce the overhead introduced by the multiplication of forest-knowledge analyses

# NOTE the multi-dimentionnal strings vector approach might not be suited to the 
# preserved analyses feature since we need to be able to determine all precedent 
# revisions of a given element, which is impossible if the revisions are not linear. 
# class RevTracker:
#     """
#     The revision tracker is a part of the advanced caching features.

#     Tracks the revision of all elements: the forest itself, trees and nodes.
#     It MUST be informed with method `increment_rev` when a transformation occurs.
#     Then it increments the relevant revisions hierachically in order to keep unrelated 
#     nodes' revision unchanged. 

#     One instance of this class can only be used to track revisions of elements 
#     in single forest, it's also assumed that the tree instances are not shared across 
#     multiple forests and all transformation are duly signaled to the instance of this class.
#     """
    
#     def __init__(self, get_anscestors: Callable[[Tree, Node], Iterable[Node]]):
#         """
#         :param get_anscestors: A callable that, given a tree and a 
#             node that lives inside it, returns the ancestors nodes
#             as a sequence from the root to the node's direct parent. 
#         """
#         self._forest_sentinel = object() # the forest object version
#         self._get_anscestors = get_anscestors # dependency injection

#         # We abuse the complex type as a two dimentional vector of ints.
#         self._revisions: MutableMapping[Element, complex] = weakref.WeakKeyDictionary()
    
#     def _elements(self, 
#                   tree: Tree | None = None, 
#                   node: Node | None = None) -> deque[object]:
#         """
#         Elements in this order: 

#             - Node
#             - Node ancestors up to the Module
#             - Tree the node lives in
#             - Forest (sentinel). 

#         The returned deque will always at least contain the fores instance. 
#         """
#         elements = deque()
#         if node:
#             assert tree is not None
#             # if there are nodes in between tree and node, add them to the stack
#             elements.extendleft(self._get_anscestors(tree, node))
#             elements.appendleft(node)
#             # elements now contains the 
#             # affected node first, then all it's ancestors.
        
#         if tree:
#             elements.append(tree)
#         elements.append(self._forest_sentinel)
#         return elements

#     def increment_rev(self, 
#                     tree: Tree | None = None, 
#                     node: Node | None = None) -> None:
#         """
#         Signals that the given tree/node has been transformed 
#         and increment it's revision hierachically. 
        
#         It's illegal to give a node without a tree. 
#         When no tree or node are given it wil be 
#         interpreted as a forest transformation.
#         """
#         rev = self._revisions
#         elements = self._elements(tree, node)
        
#         def _incr(key: object, value: complex) -> None:
#             rev.setdefault(key, 0j)
#             rev[key] += value

#         # the verion incrementation happends bottom-up, first we increment REAL part
#         # of the directly affected element, then we increment the IMAGINARY part
#         # of indirectly affected elements, up the global element version.
#         _incr(elements.popleft(), 1)
#         for e in elements:
#             _incr(e, 1j)
    
#     def rev(self, 
#                  tree: Tree | None = None, 
#                  node: Node | None = None, ) -> tuple[str, str, str]:
#         """
#         Computes the revision of the element at tree/.../node. 
#         The revision is a three layered element.
#         It's illegal to give a node without a tree. 
#         When no tree or node are given the revision of the whole 
#         forest will be returned.
#         """
#         rev = self._revisions
#         elements = self._elements(tree, node)

#         forest_rev = rev.get(elements.pop(), 0j)
#         if not elements:
#             return f"{forest_rev}", '0', '0'
        
#         tree_rev = rev.get(elements.pop(), 0j)
#         if not elements:
#             return f"{forest_rev}", f"{tree_rev}", '0'
        
#         # The version of a node is defined by the hash of it's own two dimentionned
#         # vector (IMAGINARY and REAL) combined to the REAL dimention of the enclosing
#         # elements. 
#         node_rev = rev.get(elements.popleft(), 0j)
#         parents_rev = (rev.get(e, 0j).real for e in elements)
        
#         return f"{forest_rev}", f"{tree_rev}", '/'.join(map(str, chain(parents_rev, [node_rev])))


# OLD CODE BELOW

# This code implemented a key-value based caching where the key is a tuple with a bunch 
# of meta informations including: 
#  the pass instance, 
#  the knowledge of the pass, 
#  whether the pass is complete
#  the three wheren that pass has been run
#  the node onto which the pass has been run
#  
#  the problem with that strategy is that we had to search different
#  cahed results based on meta-informations-based filters, and that complexified
#  the code...

# CACHE_KEYS = OrderedSet(
#     (   "passe",  # PassInstance
#         "knowledge",  # integer: FOREST / TREE / NODE
#         "completeness",  # boolean
#         "tree",  # Tree or None
#         "node",  # AnyNode or None
#     ))

# type _CacheKeyT = tuple[
#     PassInstance, int, bool, "Hashable | None", "Hashable | None"
# ]
# @attrs.frozen(slots=True)
# class PassManagerCache:
#     """
#     Wraps the generic L{Cache} class for caching L{CompletedPass} instances.
#     """

#     _cache: Cache[_CacheKeyT, CompletedPass]
#     tracker: RevTracker

#     def get(self, passe: PassInstance, pointer: _ElementPath) -> CompletedPass | None:
#         for k in self._mk_cache_keys_to_get_result(passe, pointer):
#             if result := self._cache.get(k):
#                 # the result is cached :)
#                 return result
#         return None

#     def set(self, result: CompletedPass):
#         key = self._mk_cache_key_to_set_result(result)
#         self._cache.set(key, result)

#     @staticmethod
#     def _mk_cache_key_to_set_result(result: CompletedPass) -> _CacheKeyT:
#         pointer: _ElementPath = result.pointer
#         # we do not use the forest part of the pointer here
#         path = pointer + (None,) * (3 - len(pointer))
#         completeness = True
#         return result.passe, result.knowledge, completeness, path[1], path[2]

#     @staticmethod
#     def _mk_cache_keys_to_get_result(
#         passe: PassInstance,
#         pointer: _ElementPath,
#     ) -> Iterator[_CacheKeyT]:
#         # we do not use the forest part of the pointer here
#         path = pointer + (None,) * (3 - len(pointer))
#         p1 = path[1]
#         p2 = path[2]
#         runs_on = int(passe.proto.runs_on)

#         yield passe, runs_on, True, p1, p2,
#         while runs_on < Level.FOREST:
#             runs_on += 1
#             yield passe, runs_on, True, p1, p2,
#         # The completeness can only be False for forest knowledge analyses.
#         yield passe, runs_on, False, p1, p2,

#     # There is too much boilerplate code around the cache management...
#     def remove(self, key: _CacheKeyT) -> None:
#         self._cache.remove(key)
#     remove.__doc__ = Cache.remove.__doc__
    
#     def search(self, **key) -> Collection:
#         return self._cache.search(**key)
#     search.__doc__ = Cache.search.__doc__
    
#     def allkeys(self) -> Collection: # used for testing
#         return self._cache.allkeys()
#     allkeys.__doc__ = Cache.allkeys.__doc__
