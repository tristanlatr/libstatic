
from __future__ import annotations

from collections import defaultdict
import dataclasses
from typing import Hashable, TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from typing import NoReturn
    from ._passmanager import Analysis
    from ._modules import ModuleCollection, Module, AnyNode

from .events import EventDispatcher, ModuleAddedEvent, ModuleRemovedEvent


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

    def set(self, analysis: type[Analysis], node: Hashable, result: AnalysisResult):
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
    
    def get(self, analysis: type[Analysis], node: Hashable) -> AnalysisResult | None:
        module = self.__modules[node]
        cache = self.__data[module]
        return cache.get(analysis, node)
    
    def set(self, analysis: type[Analysis], node: Hashable, result: AnalysisResult) -> None:
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

