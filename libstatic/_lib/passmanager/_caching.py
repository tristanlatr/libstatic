
from __future__ import annotations

from collections import defaultdict
import dataclasses
from functools import reduce
import itertools
import operator
from typing import Collection, Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from ._passmanager import Analysis
    from ._modules import ModuleCollection, Module, AnyNode

from .events import EventDispatcher, ModuleAddedEvent, ModuleRemovedEvent, ModuleTransformedEvent

from beniget.beniget import ordered_set # type: ignore

class AnalysisResult:
    """
    Simple wrapper for the result of an analysis.
    """
    @property
    def result(self) -> object:
        raise NotImplementedError(self.result)

    @classmethod
    def Success(self, result: object) -> AnalysisResult:
        return _AnalysisSuccess(result)

    @classmethod
    def Error(self, exception: Exception) -> AnalysisResult:
        return _AnalysisError(exception)
    
    def isError(self) -> bool:
        return isinstance(self, _AnalysisError)


@dataclasses.dataclass(frozen=True)
class _AnalysisError(AnalysisResult):
    _error: Exception

    @property
    def result(self) -> object:
        raise self._error


@dataclasses.dataclass(frozen=True)
class _AnalysisSuccess(AnalysisResult):
    _result: object

    @property
    def result(self) -> object:
        return self._result
    

CacheKeyLabel = ('isInterModule', 'isComplete', 'analysis', 'module', 'node')
DoNotIndexKeyLabel= ordered_set(('node',))
CacheKeyLabelSet = ordered_set(CacheKeyLabel)
CacheKey = tuple[bool, bool, type['Analysis'], 'Module', Hashable]
"""
The keys are:

- 'isInterModule': whether the analyses depends on other modules analyses
- 'isComplete': whether the analysis has a final complete result
- 'analysis': the analysis type
- 'module': the module used as context to run the analysis
- 'node': the node on which the analysis is ran, not indexed.
"""

class Cache:
    """
    This cache is implemented with a dictionary that maps a series of keys to a single value and
    an index that help searching for matching cached results.
    """
    
    class Index:
        """
        An index for cached values. The goal of this object is to give quick answer to questions like:
            - which analysis results belong to a certain module? 

        """
    
        def __init__(self) -> None:
            self.__store: dict[str, dict[Hashable, set[CacheKey]]] = defaultdict(lambda: defaultdict(ordered_set))

        def _add(self, key: CacheKey) -> None:
            # O(1)
            for label, value in zip(CacheKeyLabel, key):
                if label in DoNotIndexKeyLabel:
                    continue
                self.__store[label][value].add(key)
        
        def _discard(self, key: CacheKey) -> None:
            # O(1)
            for label, value in zip(CacheKeyLabel, key):
                if label in DoNotIndexKeyLabel:
                    continue
                self.__store[label][value].discard(key)
                # TODO: Is it worth it to delete empty sets from the structure ?
                # This would optimize kvalues() so tat we don't have to return a new set
                # but kvalues is only used in the tests at this time...

        def search(self, **key: Hashable) -> Collection[CacheKey]: # typed as Collection so it cannot be mutated.
            # O(min(len(s) for s in set of keys)) or O(1) if only one key is provided
            # Verify no junk parmeters.
            if not key:
                raise TypeError(f'Excepted at least one keyword argument')
            if not all(invalid:=(k in CacheKeyLabelSet) and (invalid:=k not in DoNotIndexKeyLabel) for k in key):
                raise TypeError(f'Invalid keyword: {invalid}')
            
            sets = [self.__store[label][value] for label, value in key.items()]
            # Fast track if only one key is provided
            if len(sets) == 1:
                return sets[0]
            # Create the intersection of sets starting with the smallest for performance reasons.
            sets.sort(key=len)
            return reduce(operator.and_, sets)
        
        def kvalues(self, key: str) -> Collection[Hashable]:
            # Verify no junk parmeters.
            if key not in CacheKeyLabel or key in DoNotIndexKeyLabel:
                raise TypeError(f'Invalid key: {key}')
            # Ignore empty sets.
            return ordered_set(k for k,v in self.__store[key].items() if v)
    
    def __init__(self) -> None:
        self.__store: dict[CacheKey, AnalysisResult] = {}
        self.index = Cache.Index()

    def add(self, key:CacheKey, value:AnalysisResult) -> None:
        # O(1)
        self.__store[key] = value
        self.index._add(key)
    
    def discard(self, key:CacheKey) -> None:
        # O(1)
        del self.__store[key]
        self.index._discard(key)

    def get(self, key:CacheKey) -> AnalysisResult | None:
        # O(1)
        return self.__store.get(key)

def make_key(modules: ModuleCollection, 
                 analysis_type: type[Analysis], 
                 node: Hashable, *, 
                 isComplete: bool) -> CacheKey:
    #O(1)
    return (
        analysis_type.isInterModules(), 
        isComplete, 
        analysis_type, 
        modules[node],
        node,
    )

class CacheProxy:
    """
    Facilitates the event-based messages with the underlying cache.
    """
   
    def __init__(self, modules: ModuleCollection, dispatcher: EventDispatcher) -> None:
        super().__init__()
        self.__modules = modules
        self.__cache = Cache()

        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)
        dispatcher.addEventListener(ModuleTransformedEvent, self._onModuleTransformedEvent)
    
    def get(self, analysis: type[Analysis], node: Hashable) -> AnalysisResult | None:
        keys = []
        options = [itertools.product(('isComplete',), (True, False))]
        for d in itertools.product(*options):
            keys.append(make_key(self.__modules, analysis, node, **dict(d)))
        try:
            return next(filter(None, (self.__cache.get(k) for k in keys)))
        except StopIteration:
            return None
    
    def set(self, analysis: type[Analysis], 
            node: Hashable, 
            result: AnalysisResult, 
            isComplete: bool = False) -> None:
        key = make_key(self.__modules, analysis, node, isComplete=isComplete)
        self.__cache.add(key, result)
   
    def analyses(self) -> Collection[Hashable]:
        return self.__cache.index.kvalues('analysis')
    
    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        # O(number of incomplete inter-modules analyses)
        # remove all results of inter-modules analyses that are not complete
        for k in self.__cache.index.search(isInterModule=True, isComplete=False):
            self.__cache.discard(k)

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        # O(number of analyses belonging to that module + number of inter-modules analyses)
        # this is less optimized since it should be relatively uncommon to remove a module from the system.

        # remove all results that are indexed in that module
        for k in self.__cache.index.search(module=event.mod):
            self.__cache.discard(k)
        # remove all inter-modules analyses
        # TODO: filter out the ones that do not depend on the removed module,
        # this require the import graph to be considered as a special analysis like
        # the ancestors and the root module mapping..

        for k in self.__cache.index.search(isInterModule=True):
            self.__cache.discard(k)
    
    def _onModuleTransformedEvent(self, event: ModuleTransformedEvent):
        """
        The given module has been transformed, this is automatically called
        at the end of a transformation if it updated the module.
        """
        # the transformation updated the AST, so analyses may need to be rerun
        # Instead of clearing the entire cache, only invalidate analysis that are affected
        # by the transformation.
        
        preserves = event.transformation.preservesAnalyses
        discard = self.__cache.discard
        
        # stuff that needs to be invalidated: 
        # - all inter-modules analyses except few preserved
        # - all intra-module analyses belonging to a given module except few preserved
        intra_local_module = self.__cache.index.search(isInterModule=False, module=event.mod)
        inter_modules = self.__cache.index.search(isInterModule=True)
        # TODO: filter out the inter-modules analysis that does not depend on the transformed module.

        for key in itertools.chain(intra_local_module, inter_modules):
            _, _, analysis, _, _ = key
            # TODO: might be smart to cast 'preserves' to set but need to see if the analysis like patterns will still work.
            if analysis in preserves: 
                continue
            discard(key)
