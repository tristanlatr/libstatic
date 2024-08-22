
from __future__ import annotations

from collections import defaultdict
import dataclasses
from functools import partial
from typing import Any, Hashable, TYPE_CHECKING, Iterable, Iterator, Literal
from weakref import WeakSet

if TYPE_CHECKING:
    from typing import NoReturn
    from ._passmanager import Analysis
    from ._modules import ModuleCollection, Module, AnyNode

from .events import ClearAnalysisEvent, EventDispatcher, ModuleAddedEvent, ModuleRemovedEvent, ModuleTransformedEvent

from beniget.beniget import ordered_set

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

# @dataclasses.dataclass(slots=True, frozen=True)
# class Entry:
#     parent: Entry | None
#     data: dict[Hashable, Any]

# @dataclasses.dataclass(slots=True, frozen=True)
# class EntryNotFound(Exception):
#     currentEntry: Entry
#     axe: Hashable

# class Store:
#     """
#     Store items arround a series of keys.
    
#     For each key, an item should only have one value associated: 
#     like a point, except we can use an arbitrary number of axes.
    
#     >>> class shirts:...
#     >>> i = Store(['kind', 'price', 'floor'])
#     >>> shirt, shorts, jaquet = shirts(), object(), object()
#     >>> i.set(shirt, price=7, floor=1, kind='top')
#     >>> i.set(shorts, price=14, floor=-1, kind='bottom')
#     >>> i.set(jaquet, price=30, floor=0, kind='top')
#     >>> i._data
#     {'top': {7: {1: {'__value__': <...shirts...>}}, 30: {0: {'__value__': <object...>}}}, 'bottom': {14: {-1: {'__value__': <object...>}}}}
#     >>> i.get(kind='top', price=7, floor=1)
#     <...shirts object at ...>
#     >>> i.clear(kind='top')
#     >>> i._data
#     {'bottom': {14: {-1: {'__value__': <object...>}}}}
#     >>> print(i.get(kind='top', price=7, floor=1))
#     None
#     >>> i.set(shirt, price=7, floor=1, kind='top')
#     >>> i._data
#     {'bottom': {14: {-1: {'__value__': <object...>}}}, 'top': {7: {1: {'__value__': <...shirts...>}}}}
#     >>> i.clear(kind='top', price=6) # there is nothing at top/price 6
#     >>> i._data
#     {'bottom': {14: {-1: {'__value__': <object...>}}}, 'top': {7: {1: {'__value__': <...shirts...>}}}}
#     >>> i.get(kind='top', price=7, floor=1)
#     <...shirts object at ...>
#     >>> i.clear(kind='top', price=7, floor=2) # there is nothing at top/price 7/floor 2
#     >>> i._data
#     {'bottom': {14: {-1: {'__value__': <object...>}}}, 'top': {7: {1: {'__value__': <...shirts...>}}}}
#     >>> i.get(kind='top', price=7, floor=1)
#     <...shirts object at ...>
#     >>> i.clear(kind='top', price=7)
#     >>> print(i.get(kind='top', price=7, floor=1))
#     None
#     >>> i._data
#     {'bottom': {14: {-1: {'__value__': <object...>}}}, 'top': {}}
#     >>> print(i.get(kind='bottom', price=14, floor=-1))
#     <object...>

#     """
#     def __init__(self, axes: Iterable[str]):
#         """
#         @param axes: List of fields that the contained objects are indexed on. 
#         """
#         self.__axes = tuple(axes)
#         self.__sortedAxes = sorted(axes)
#         self._data = {}
#         # self.__contains = WeakSet()
    
#     def __verifyComplete(self, **axes: Hashable):
#         if not sorted(axes) == self.__sortedAxes:
#             raise TypeError('missing keyword parameters: '
#                             f'{sorted(set(self.__axes) - set(axes))}')
    
#     def __getEntry(self, create:bool, /, 
#                    **axes: Hashable) -> Entry:
#         e = Entry(None, self._data)
#         for axe in self.__axes:
#             try:
#                 value = axes.pop(axe)
#             except KeyError:
#                 if axes: # At his point axes MUST be empy or esle we raise an exeption
#                     raise TypeError(f'inconsistent usage of the indexer, some axes remains: {axes}, maybe missing keyword: {axe}')
#                 break # can't happend if __verifyComplete is called.
#             if value not in e.data:
#                 if create:
#                     e.data[value] = {}
#                 else:
#                     raise EntryNotFound(e, value)
#             e = Entry(e, e.data[value])
        
#         return e
    
#     def set(self, item: Hashable, **axes: Hashable):
#         self.__verifyComplete(**axes)
#         self.__getEntry(True, **axes).data['__value__'] = item

#     def get(self, **axes: Hashable):
#         self.__verifyComplete(**axes)
#         try:
#             # Look for an exact match.
#             return self.__getEntry(False, **axes).data['__value__']
#         except EntryNotFound as e:
#             curr = e.currentEntry
#             while True:
#                 # Look for a region match
#                 try:
#                     return curr.data['__region__']
#                 except KeyError:
#                     if curr.parent is None:
#                         break
#                     curr = curr.parent
#             raise KeyError(e.axe) from e

#     def clear(self, **axes: Hashable):
#         """
#         Actually deletes the region described by the axes. 
#         The axes doesn't have to be full, in this case a whole region will be cleared.
#         """
#         # actually delete the data from the store\
#         k_to_del = axes.pop(self.__axes[-1])
#         try:
#             store = self.__getEntry(False, **axes)
#         except KeyError:
#             pass
#         else:
#             try:
#                 del store[k_to_del]
#             except KeyError:
#                 pass



class AnalysisCache:
    """
    A single module's cache.
    """
    # """
    # The strucutre of the cache consist in nested dicts.
    # But this class facilitates the messages with the module pass manager.
    # """

    def __init__(self) -> None:
        self.__data: dict[type[Analysis], dict[Hashable, AnalysisResult]] = {}

    def set(self, analysis: type[Analysis], 
            node: Hashable, 
            result: AnalysisResult):
        """
        Store the analysis result in the cache.
        """
        if analysis.doNotCache: raise RuntimeError()
        if analysis not in self.__data:
            self.__data[analysis] = {}
        self.__data[analysis][node] = result

    def get(self, analysis: type[Analysis], node: Hashable) -> AnalysisResult | None:
        """
        Query for the cached result of this analysis.
        """
        if analysis in self.__data:
            try:
                return self.__data[analysis][node]
            except KeyError:
                return None
        return None

    def clear(self, analysis: type[Analysis] | None):
        """
        Get rid of the the given analysis result, this will 
        clear the result for all nodes in the cache's module.

        Clear all analyses if None is passed.
        """
        if analysis is None:
            self.__data.clear()
        elif analysis in self.__data:
            del self.__data[analysis]

    def analyses(self) -> Iterator[type[Analysis]]:
        """
        Get an iterator on all analyses types in this cache.
        """
        yield from self.__data
    
    def isEmpty(self) -> bool:
        return not self.__data

class AnalyisCacheProxy:
    """
    The cache for several modules.
    """
   
    def __init__(self, modules: ModuleCollection, dispatcher: EventDispatcher) -> None:
        super().__init__()
        self.__modules = modules
        self.__data: dict[Module, AnalysisCache] = {}

        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        dispatcher.addEventListener(ClearAnalysisEvent, partial(_onClearAnalysisEvent, self))
        dispatcher.addEventListener(ModuleTransformedEvent, partial(_onModuleTransformedEvent, self, dispatcher))
        dispatcher.addEventListener(ModuleAddedEvent, partial(_onModuleAddedOrRemovedEvent, self, dispatcher))
        dispatcher.addEventListener(ModuleRemovedEvent, partial(_onModuleAddedOrRemovedEvent, self, dispatcher))

    
    def get(self, analysis: type[Analysis], node: Hashable) -> AnalysisResult | None:
        module = self.__modules[node]
        cache = self.__data[module]
        return cache.get(analysis, node)
    
    def set(self, analysis: type[Analysis], 
            node: Hashable, 
            result: AnalysisResult, 
            isComplete: bool = False) -> None:
        module = self.__modules[node]
        cache = self.__data[module]
        cache.set(analysis, node, result)
    
    def clear(self, analysis: type[Analysis] | None, mod: AnyNode | None) -> None:
        if mod is None:
            for cache in self.__data.values():
                cache.clear(analysis)
        else:
            module = self.__modules[mod]
            cache = self.__data[module]
            cache.clear(analysis)

    def modulesAnalyses(self) -> Iterator[tuple[Module, type[Analysis]]]:
        """
        For each analyses in each module's cache, yield C{(Module, type[Analysis])}.
        """
        for mod, cache in self.__data.items():
            for analysis in tuple(cache.analyses()): # tuple() avoids RuntimeError: dictionary changed size during iteration
                yield mod, analysis
    
    def analyses(self) -> Iterator[type[Analysis]]:
        """
        Iter on all analyses types.
        """
        # TODO: Maybe having a separate cache for inter modules analyses would improve performance cost induced by
        # iterating over all analyses
        r = set()
        for _, a in self.modulesAnalyses():
            if a in r:
                continue
            r.add(a)
            yield a
    
    def _data(self) -> dict[Module, AnalysisCache]:
        return self.__data
    
    def _merge(self, other: AnalyisCacheProxy) -> None:
        # At this point, this only works if the other cache only contains information about modules that
        # this cache doesn't know about. This is a private API. 
        self.__data.update(other._data())
    
    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        self.__data[event.mod] = AnalysisCache()

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        del self.__data[event.mod] # this **should** free the memory from all analyses in the cache for this module.

def _onModuleAddedOrRemovedEvent(cache: AnalyisCacheProxy, dispatcher: EventDispatcher,
                                 event: ModuleRemovedEvent | ModuleAddedEvent):
    """
    Clear the analysis cache of all inter-modules analyses. 
    # TODO: This could use some fine tuning to mark the results as preserved.
    """
    for a in cache.analyses():
        if a.isInterModules():
            dispatcher.dispatchEvent(
                ClearAnalysisEvent(a, event.mod.node)
            )


def _onModuleTransformedEvent(cache: AnalyisCacheProxy, dispatcher: EventDispatcher,
                              event: ModuleTransformedEvent):
    """
    Alert that the given module has been transformed, this is automatically called
    at the end of a transformation if it updated the module.
    """
    transformation = event.transformation
    mod: Module = event.mod
    # the transformation updated the AST, so analyses may need to be rerun
    # Instead of clearing the entire cache, only invalidate analysis that are affected
    # by the transformation.
    invalidated_analyses: set[type[Analysis]] = ordered_set()
    # TODO: The goal would be to avoid iterating accross analyses whenever a module is transformed.
    # - We could imagine that the cache keeps tracls of dirty results in an optmized structure
    # and re-generate as needed only.
    # We could al imagine a world where determining which analyses to delete is done in o(1). 
    # here is the matrix's keys: 'isInterModule', 'isComplete', 'analysis', 'module', 'node', '__value__'
    
    # stuff that needs to be invalidated: 
    # - all inter-modules analyses
    # - all inter-modules analyses that are not complete
    # - all intra-module analyses belonging to a given module

    # - all inter-modules analyses expect few preserved
    # - all inter-modules analyses that are not complete expect few preserved
    # - all intra-module analyses belonging to a given module expect few preserved

    for m, analysis in cache.modulesAnalyses():
        if (
            # if the analysis is explicitely presedved by this transform,
            # do not invalidate.
            (analysis not in transformation.preservesAnalyses)
            and (
                # if it's not explicitely preserved and the transform affects the module
                # invalidate.             or if the analysis requires other modules
                (m.node is mod.node) or (analysis.isInterModules())
            )
        ):
            invalidated_analyses.add(analysis)

    for analys in invalidated_analyses:
        # alert that this analysis has been invalidated
        dispatcher.dispatchEvent(
            ClearAnalysisEvent(analys, mod.node)
        )


def _onClearAnalysisEvent(cache: AnalyisCacheProxy, event: ClearAnalysisEvent):
    """
    Clear the cache from this analysis.
    """
    analysis = event.analysis

    if analysis.isInterModules():
        cache.clear(analysis, None)
    else:
        cache.clear(analysis, event.node)

