"""
This module provides classes and functions for pass management.

There are two kinds of passes: transformations and analysis.
    * ModuleAnalysis, FunctionAnalysis and NodeAnalysis are to be
      subclassed by any pass that collects information about the AST.
    * gather is used to gather (!) the result of an analyses on an AST node.
    * Transformation is to be sub-classed by any pass that updates the AST.
    * apply is used to apply (sic) a transformation on an AST node.
"""

# TODO: Things to change:
# Concerning the cache: 

# 1 - Only true analysis results should be in there
# adapters do not belong in the cache since they can be re-created on the fly

# 2 - The data should only be cached at one place:
# ScopeToModuleAnalysisResult should not store the data but always delegate to gather:
# this way we don't have to register the mapping proxy as listener to the invalidated analysis event
# since it's not storing any data.
# the AllModulesAnalysis class is also to be refactored to solve the issue of the mapping-like analysis values
# should adopt a single and clear manner to aggregate module analysis results and it is not to do anything special
# when the result is a dict. Adapters can then be created to act like a mapping for the underlying 

#  -> There is no such things as a analysis that runs on all modules
#       This will be called a analysis adaptor: these objects are composed by the PassManager and redirect
#       all requests to the underlying pm.
# - PassManager should rely on lower level ModulePassManager that handles everything
# at the single module level, offering an pythran-like interface. 
# 
# Questions: Should we have two caches, one for intra-modules analyses one for inter-modules analyses ?
# One PassManager is then composed by several ModulePassManagers that actually run all the analyses. 
# How to cache transformation results ? -> no need at the moment since the analyses results can be 
# cached and the transform time will be very quick anyway
#
# Pass instrumentation: Adding callbacks before and after passes: 
# Use this as an extensions system: 
# - First application is to gather statistics about analyses run times
# - Other applications includes generating a tree hash before and after each transform

# about caching analysis results in files
# - all modules should have hash keys generated and updated on transformations:
#       1/ local hash (module ast + modulename + is_package, ...): that is only valid for intra-module analysis
#       2/ inter-modules hash: that is a combinaison of the module hash and all it's dependencies'inter-module hashes.

# If an analysis raises an exception during processing, this execption should be saved in the cache so it's re-raised
# whenever the same analysis is requested again.

# Ressources: 
# LLVM PassManager documentation: https://llvm.org/doxygen/classllvm_1_1PassManager.html
from __future__ import annotations

from collections import defaultdict
import enum
from functools import lru_cache, partialmethod, partial
import itertools
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Hashable, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar, Generic, Sequence, Type, Union, TypeAlias, final, overload
import ast
from time import time
from contextlib import contextmanager
from pathlib import Path
import dataclasses
import abc

from beniget.ordered_set import ordered_set

from .exceptions import NodeLocation

@lru_cache(maxsize=None, typed=True)
def partialclass(cls: Type[Any], *args: Any, **kwds: Any) -> Type[Any]:
    """
    Bind a class to be created with some predefined __init__ arguments.
    """
    class NewPartialCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds) #type: ignore
        __class__ = cls
    assert isinstance(NewPartialCls, type)
    return NewPartialCls


def location(node:ast.AST, filename:'str|None'=None) -> str:
    return str(NodeLocation.make(node, filename=filename))

if TYPE_CHECKING:
    _FuncOrClassTypes = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    _ScopesTypes = _FuncOrClassTypes | ast.SetComp | ast.DictComp | ast.ListComp | ast.GeneratorExp | ast.Lambda
    _CannotContainClassOrFunctionTypes = (ast.expr | ast.Return | ast.Delete |
            ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Raise | ast.Assert | 
            ast.Import | ast.ImportFrom | ast.Global | ast.Nonlocal | ast.Expr |
            ast.Pass | ast.Break | ast.Continue)
            
else:
    _FuncOrClassTypes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    _ScopesTypes = (*_FuncOrClassTypes, ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Lambda)
    _CannotContainClassOrFunctionTypes = (ast.expr, ast.Return, ast.Delete, 
            ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Raise, ast.Assert, 
            ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Expr, 
            ast.Pass, ast.Break, ast.Continue)

# TODO add ast.TypeAlias to the list when python 3.13 is supported by gast

def walk(node:ast.AST, typecheck:type|None=None, stop_typecheck:type|None=None):
    """
    Recursively yield all nodes matching the typecheck 
    in the tree starting at *node* (excluding *node* itself), in bfs order.

    Do not recurse on children of types matching the stop_typecheck type.

    See also `ast.walk` that behaves diferently.
    """
    from collections import deque
    todo = deque(ast.iter_child_nodes(node))
    while todo:
        node = todo.popleft()
        if stop_typecheck is None or not isinstance(node, stop_typecheck):
            todo.extend(ast.iter_child_nodes(node))
        if typecheck is None or isinstance(node, typecheck):
            yield node


class Event:
    """
    Base event to use with EventDispatcher.
    """

EventListener = Callable[[Event], None]

class EventDispatcher:
    """
    Generic event dispatcher which listen and dispatch events
    """

    def __init__(self):
        self._events: dict[type[Event], list[EventListener]] = dict()

    def has_listener(self, event_type: type[Event], listener: EventListener) -> bool:
        """
        Return true if listener is register to event_type
        """
        # Check for event type and for the listener
        if event_type in self._events:
            return listener in self._events[event_type]
        else:
            return False

    def dispatch_event(self, event: Event) -> None:
        """
        Dispatch an instance of Event class
        """
        if __debug__:
            D = True
        else:
            D = False
        
        if D: print(f'dispatching event {event!r}')
        # Dispatch the event to all the associated listeners
        if type(event) in self._events:
            listeners = self._events[type(event)]

            for listener in listeners:
                if D: print(f'dispatching event {event!r} to listener {listener!r}')
                listener(event)

    def add_event_listener(self, event_type: type[Event], listener: EventListener) -> None:
        """
        Add an event listener for an event type
        """
        # Add listener to the event type
        if not self.has_listener(event_type, listener):
            listeners = self._events.get(event_type, [])
            listeners.append(listener)
            self._events[event_type] = listeners

    def remove_event_listener(self, event_type: type[Event], listener: EventListener) -> None:
        """
        Remove event listener.
        """
        # Remove the listener from the event type
        if self.has_listener(event_type, listener):
            listeners = self._events[event_type]

            if len(listeners) == 1:
                # Only this listener remains so remove the key
                del self._events[event_type]

            else:
                # Update listeners chain
                listeners.remove(listener)

                self._events[event_type] = listeners

@dataclasses.dataclass(frozen=True)
class ModuleChangedEvent(Event):
    """
    When a module is transformed.
    """
    mod: 'Module'

@dataclasses.dataclass(frozen=True)
class InvalidatedAnalysisEvent(Event):
    """
    When an analysis is invalidated.
    """
    analysis: type[Analysis]
    node: ast.Module

@dataclasses.dataclass(frozen=True)
class ModuleAddedEvent(Event):
    """
    When a module is added to the passmanager.
    """
    mod: 'Module'

@dataclasses.dataclass(frozen=True)
class ModuleRemovedEvent(Event):
    """
    When a module is removed from the passmanager.
    """
    mod: 'Module'


@dataclasses.dataclass(frozen=True)
class SearchContext:
    """
    A fake sys.path to search for modules. See typeshed.
    """


@dataclasses.dataclass(frozen=True)
class Module:
    """
    The specifications of a python module.
    """
    
    node: ast.Module
    modname: str

    filename: str | None = None
    is_package: bool = False
    is_namespace_package: bool = False
    is_stub: bool = False
    code: str | None = None


class Finder:
    """
    In charge of finding python modules and creating ModuleSpec instances for them.
    """
    
    search_context: SearchContext
    
    def find_module_by_name(self, modname: str) -> Module:
        ...
    
    def iter_module_path(self, path: Path) -> Iterator[Module]:
        ...


class ModuleCollection(Mapping[str|ast.Module|ast.AST, Module]):
    """
    A fake sys.modules to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher, roots: RootModuleMapping):
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[ast.Module, Module] = {}
        self.__roots = roots

        dispatcher.add_event_listener(ModuleAddedEvent, self._add)
        dispatcher.add_event_listener(ModuleRemovedEvent, self._remove)
    
    def _add(self, event: ModuleAddedEvent):
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if self.get(modname) not in (None, mod):
            raise ValueError(f'a module named {modname!r} '
                             f'already exist: {self[modname]}')

        if self.get(modnode) not in (None, mod):
            raise ValueError(f'the ast of the module {modname!r} is already '
                             f'associated with another module: {self[modnode]}')

        # register the module as beeing a part of this collection.
        self.__name2module[modname] = mod
        self.__node2module[modnode] = mod
    
    def _remove(self, event: ModuleRemovedEvent):
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if modname not in self or modnode not in self:
            raise ValueError(f'looks like this module is not in the collection: {mod}')
        
        # remove the module from the collection
        del self.__name2module[modname]
        del self.__node2module[modnode]
    
    # mapping interface

    def __getitem__(self, __key: str|ast.Module|ast.AST) -> Module:
        if isinstance(__key, str):
            return self.__name2module[__key]
        elif isinstance(__key, ast.Module):
            return self.__node2module[__key]
        elif isinstance(__key, ast.AST):
            return self.__node2module[self.__roots[__key]]
        else:
            raise TypeError(f'unexpected key type: {__key}')
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.__name2module)

    def __len__(self) -> int:
        return len(self.__name2module)

# class AnalysisCommand:
#     """
#     Container for running and re-runnin an analysis.
#     """

#     pm: 'PassManager'
#     analysis: 'Analysis'
#     node: ast.AST

#     def run(self):
#         ...

@dataclasses.dataclass(frozen=True)
class CachedResult:
    result: object

class PassManagerCache:
    """
    The strucutre of the cache consist in nested dicts. 
    But this class facilitates the messages with the pass manager.
    """
    def __init__(self, root_modules: Mapping[ast.AST, ast.Module]) -> None:
        self._data: 'dict[type[Analysis], dict[ast.AST | None, CachedResult]]' = defaultdict(dict)
        """
        Dict from analysis type to dict from node to CachedResult.
        """

        self._modmap: 'dict[ast.Module, dict[type, list[ast.AST]]]' = defaultdict(dict)
        """
        Dict from module to list of tuples(analysis type, node).
        Only valid non-modules and non-adapter analyses. This helps keeping track of which analysis
        exist for a given module. Because if a module is invalidated, all the analysis of the ast
        of this module becomes invalid, so we need a manner to iterate thu all the results in a module.
        """

        self._root_modules = root_modules
    
    def set(self, analysis: Hashable, node: Hashable, result: CachedResult):
        """
        Store the analysis result in the cache.

        When the analysis type is not a module analysis or an adapter
        """        
        self._data[analysis][node] = result
        
        if isinstance(node, ast.AST):
            # rules: 
            # - an adaptor analysis can only be ran on Projecy or ast.Modules
            # - other analysis must be ran on ast nodes.
            mod = self._root_modules[node] if not isinstance(node, ast.Module) else node
            self._modmap[mod].setdefault(analysis, []).append(node)

    def get(self, analysis: type, node: ast.AST | None) -> Optional[CachedResult]:
        """
        Query for the cached result of this analysis.
        """
        if analysis in self._data and node in self._data[analysis]:
            return self._data[analysis][node]
        return None
    
    def clear(self, analysis: type[Analysis], node: ast.Module):
        """
        Get rid of the the given analysis result.
        """

        if __debug__:
            D = True
        else:
            D = False

        if issubclass(analysis, AnalysisAdaptor):
            raise ValueError(f"can't clear andaptor {analysis.__qualname__!r} from the cache, "
                             "you need to clear the underlying analysis")

        if analysis in self._data and node in self._data[analysis]:
            # delete the data if it's a module analysis
            if D: print(f'Clearing analysis {analysis} for node {node}')
            del self._data[analysis][node]
            self._modmap[node][analysis].remove(node)

        # We don't bother delete empty list and dicts at the moment.
        # So at his time, the analysis might be a node, function or class analysis.
        # issubclass(analysis, (ClassAnalysis, FunctionAnalysis, NodeAnalysis)) and  
        if analysis in self._data:
            analysis_data = self._data[analysis]
            # need to clear all results belonging to the given module
            for n in self._modmap[node].get(analysis, ()):
                if n in analysis_data:
                    # delete the data if it's a module analysis
                    if D: print(f'Clearing analysis {analysis} for node {n}')
                    del analysis_data[n]
                    self._modmap[node][analysis].remove(n)

    def iter_analysis_types(self) -> Iterable[type[Analysis]]:
        return self._data.keys()


T = TypeVar('T')
RunsOnT = TypeVar('RunsOnT')
ReturnsT = TypeVar('ReturnsT')
ReturnsMapT = TypeVar('ReturnsMapT', bound=Mapping)
ChildAnalysisReturnsT = TypeVar('ChildAnalysisReturnsT')


class BasePass(Generic[RunsOnT, ReturnsT], abc.ABC):

    dependencies: Sequence[type[BasePass]] = ()
    """
    Statically declared dependencies
    """

    def attach(self, pm: PassManager):
        self.passmanager = pm

    def verify_dependencies(self):
        """
        1. Checks no analysis are called before a transformation,
           as the transformation could invalidate the analysis.
        2. Unallow function analysis as part of a class analysis dependencies
           Unallow class analysis as part of a function analysis dependencies
           Unallow function analysis as part of a node analysis dependencies
           Unallow class analysis as part of a node analysis dependencies;
           Use a module analsis adapter for all of those instead.
        """    
        has_analysis = False
        for d in self.dependencies:
            if not has_analysis and isinstance(d, Analysis):
                has_analysis = True
            elif has_analysis and isinstance(d, Transformation):
                raise ValueError(f"invalid dependencies order for {self.__class__}")

        sorted_dependencies = sorted(self.dependencies, key=lambda d:isinstance(d, Transformation))
        if list(sorted_dependencies) != list(self.dependencies):
            raise ValueError(f"invalid dependencies order for {self.__class__}")
        
        if isinstance(self, FunctionAnalysis):
            if any(issubclass(d, ClassAnalysis) for d in self.dependencies):
                raise ValueError(f"invalid dependencies: function analysis can't have dependencies on class analysis")
        elif isinstance(self, ClassAnalysis):
            if any(issubclass(d, FunctionAnalysis) for d in self.dependencies):
                raise ValueError(f"invalid dependencies: class analysis can't have dependencies on function analysis")
        elif isinstance(self, NodeAnalysis):
            if any(issubclass(d, (FunctionAnalysis, ClassAnalysis)) for d in self.dependencies):
                raise ValueError(f"invalid dependencies: node analysis can't have dependencies on function or class analysis")

    def prepare(self, node: RunsOnT):
        '''Gather analysis result required by this analysis'''
        self.verify_dependencies()

        for analysis in self.dependencies:
            if issubclass(analysis, Transformation):
                # Transformations always run the module since there 
                # are only module wide transformations at the moment.
                self.passmanager.apply(analysis, 
                                       self.passmanager.modules[node].node)
            elif issubclass(analysis, Analysis):
                gather_on_node = node
                if issubclass(analysis, ModuleAnalysis):
                    gather_on_node = self.passmanager.modules[node].node
                elif issubclass(analysis, AllModulesAnalysis):
                    gather_on_node = Project

                # TODO: Use a descriptors for all non-transformations.
                result = self.passmanager.gather(analysis, gather_on_node) # type:ignore[var-annotated]
                setattr(self, analysis.__name__, result)
            else:
                raise TypeError(f"dependencies should be a Transformation or an Analysis, not {analysis}")

    def run(self, node: RunsOnT) -> ReturnsT:
        """Override this to add special pre or post processing handlers."""
        self.prepare(node)
        return self.do_pass(node)
    
    @abc.abstractmethod
    def do_pass(self, node: RunsOnT) -> ReturnsT:
        """
        Override this to add actual pass logic. 
        """


class Analysis(BasePass[RunsOnT, ReturnsT]):
    """
    A pass that does not change its content but gathers informations about it.
    """
    update = False # TODO: Is this needed

    def run(self, node: RunsOnT) -> ReturnsT:
        typ = type(self)
        cached_result = self.passmanager._cache.get(typ, node)
        if cached_result is not None:
            result = cached_result.result
        else:
            result = super().run(node)
            self.passmanager._cache.set(typ, node, CachedResult(result))
        return result

    def apply(self, node: RunsOnT) -> tuple[bool, RunsOnT]:
        print(self.run(node))
        return False, node


class ModuleAnalysis(Analysis[ast.Module, ReturnsT]):

    """An analysis that operates on a whole module."""


class FunctionAnalysis(Analysis[Union[ast.FunctionDef, ast.AsyncFunctionDef], ReturnsT]):

    """An analysis that operates on a function."""
    

class ClassAnalysis(Analysis[ast.ClassDef, ReturnsT]):

    """An analysis that operates on a class."""


class NodeAnalysis(Analysis[ast.AST, ReturnsT]):

    """An analysis that operates on any node."""


class AnalysisAdaptor(Analysis[RunsOnT, ReturnsT]):
    """
    An analysis built from another more-specific analysis.
    """


class AllModulesAnalysis(
    AnalysisAdaptor[None, Mapping[ast.AST, ChildAnalysisReturnsT]],
    Generic[ChildAnalysisReturnsT]):
    """
    Adapt a module analysis in order to work for all modules seemlessly. 

    The result of this analysis is a chain map of lazy mapping objects that will 
    gather the analysis results when first used.

    The result observes the ModuleCollection such that at each add_module() call 
    it will make sure the new module is a part of it as well.
    It will also observes when the child analyses are invalidated so the chain-map entry 
    is reset to non-analysed state. 

    """
    # good opportunity to use WeakKeyDictionary?

    # Implementation note: this kind of analysis just takes None as the node since it already has
    # access to the module collection through the self.passmanager.modules attribute.

    def __init__(self, analysis: type):
        self._analysis = analysis

    def run(self, node:_Project) -> Mapping[ast.AST, T]:
        assert isinstance(node, _Project)
        # TODO
        m = AllModulesAnalysisResult(self.passmanager, [])
        self.result = m
        return m

# NO FINISHED!
class AllModulesAnalysisResult(Mapping[ast.AST, T]):
    # used for module to all modules analysis promotions
    def __init__(self, passmanager: 'PassManager', 
                 maps: Sequence[Tuple[Module, Mapping]]) -> None:
        # needs another parameter in toder to use eager re-computing
        # if not based on whether the anlaysis is 'ancestors' or not.
        super().__init__()

        self._maps = []

        # build a mapping from module to mapping index.

        # register the event listener
        passmanager._dispatcher.add_event_listener(
            InvalidatedAnalysisEvent, self._invalidated_analysis
        )
        passmanager._dispatcher.add_event_listener(
            ModuleAddedEvent, self._module_added
        )
        passmanager._dispatcher.add_event_listener(
            ModuleRemovedEvent, self._module_added
        )
    
    def __getitem__(self, __key: ast.AST):
        # dertermine which module this node belongs to,
        # then is the analysis have not been ran yet, run it.
        # Basically we rely on a 
        ...
    
    def __len__(self) -> int: # from cpython
        return len(set().union(*self._maps))     # reuses stored hash values if possible

    def __iter__(self) -> Iterator: # from cpython
        d = {}
        for mapping in reversed(self._maps):
            d.update(dict.fromkeys(mapping))    # reuses stored hash values if possible
        return iter(d)

    def _invalidated_analysis(self, event:InvalidatedAnalysisEvent):
        ...
    
    def _module_added(self, event:ModuleAddedEvent):
        ...
    
    def _module_removed(self, event:ModuleRemovedEvent):
        ...


class ScopeToModuleAnalysis(
    AnalysisAdaptor[ast.Module, Mapping['_FuncOrClassTypes', ChildAnalysisReturnsT]], 
    Generic[ChildAnalysisReturnsT]):

    # the results of each analysis and make them accessible in the result dict proxy.
    # the result must watch invalidations in order to reflect the changes
    
    def __init__(self, analysis: type[Analysis[_FuncOrClassTypes, ChildAnalysisReturnsT]]):
        self.__analysis_type = analysis

    def __keys_factory(self, node: ast.Module) -> Collection[_FuncOrClassTypes]:
        if isinstance(self.__analysis_type, ClassAnalysis):
            typecheck = (ast.ClassDef,)
        elif isinstance(self.__analysis_type, FunctionAnalysis):
            typecheck = (ast.FunctionDef, ast.AsyncFunctionDef)
        else: 
            raise NotImplementedError()
        return ordered_set(walk(node, 
                                typecheck,
                                # For performance, we stop the tree waling as soon as we hit
                                # statement nodes that cannot contain class or functions.
                                _CannotContainClassOrFunctionTypes))

    def do_pass(self, node: ast.Module) -> Mapping[_FuncOrClassTypes, ChildAnalysisReturnsT]:
        assert isinstance(node, ast.Module)
        self.result = r = ScopeToModuleAnalysisResult(self.passmanager, 
                node, self.__analysis_type, self.__keys_factory)
        return r


class ScopeToModuleAnalysisResult(Mapping['_FuncOrClassTypes', ChildAnalysisReturnsT], 
                                  Generic[ChildAnalysisReturnsT]):
    # used for class/function to module analysis promotions

    def __init__(self, passmanager: PassManager, 
                 module: ast.Module, 
                 analysis_type: type[Analysis[_FuncOrClassTypes, ChildAnalysisReturnsT]],
                 keys_factory: Callable,
                 ) -> None:
    
        assert issubclass(analysis_type, (FunctionAnalysis, ClassAnalysis))

        self.__module = module
        self.__analysis_type = analysis_type
        self.__keys_factory = keys_factory

        self.__keys = keys_factory(module)
        self.__dict: dict[_FuncOrClassTypes, ChildAnalysisReturnsT] = {}

        # register the event listener
        passmanager._dispatcher.add_event_listener(
            InvalidatedAnalysisEvent, self._invalidated_analysis
        )
        self.__pm = passmanager
    
    def keys(self) -> Collection[_FuncOrClassTypes]: # type:ignore[override]
        return self.__keys
    
    def __contains__(self, __key: object) -> bool:
        return __key in self.__keys

    def __getitem__(self, __key: _FuncOrClassTypes) -> ChildAnalysisReturnsT:
        if __key not in self.__keys:
            raise KeyError(__key)
        
        if __key not in self.__dict:
            r: ChildAnalysisReturnsT
            self.__dict[__key] = r = self.__pm.gather(self.__analysis_type, __key)
            return r

        return self.__dict[__key]
    
    def __iter__(self) -> Iterator[_FuncOrClassTypes]:
        return iter(self.__keys)
    
    def __len__(self) -> int:
        return len(self.__keys)

    def _invalidated_analysis(self, event: InvalidatedAnalysisEvent) -> None:
        analysis, node = event.analysis, event.node
        # if the analysis matches the self._analysis_type and the node is the module.
        if analysis is self.__analysis_type and node is self.__module:
            self.__keys = self.__keys_factory(self.__module)
            self.__dict = {}

# Where to put the code that coerces the result into a dict, the dict should be considered the result?
# 

# @lru_cache(maxsize=None)
# def AdaptForAllModules(analysis) -> AllModulesAnalysis:
#     """
#     Create an adapter type for the given analysis that will return a lazy mapping view
#     of all modules result in the pass manager.
#     """
#     return partialclass(AllModulesAnalysis, analysis)
    

# @lru_cache(maxsize=None)
# def AdaptForOneModule(analysis) -> ScopeToModuleAnalysis:
#     """
#     Create an adapter type for the given function or class analysis that will return
#     a mapping from node to the analysis result.
#     """
#     return partialclass(ScopeToModuleAnalysis, analysis)


# TODO: Work for having function and classes transformations 
# that will invalidate only the required function analyses.
# Seulement les transformations de modules peuvent changer l'interface
# publique d'un module (ex changer signature d'une fonction our autre).
# Une transformation d'un certain niveau ne doit pas interferrer avec 
# les aures transformation du même niveau.
# Exemple: Pas possible d'ajouter des global ou nonlocal keyword depuis 
# une transformation de fonction.
# Compatibilité avec NodeTransformer + NodeVisitor pour les analyses.
# Comment determiner si la pass update l'ast: calculer un hash avant et apres. 
class Transformation(BasePass[ast.Module, ast.Module]):
    """A pass that updates its content."""
    
    preserves_analysis = ()
    """
    This variable should be overridden by subclasses to provide specific list.
    By default, assume all analyses are affected. 
    """

    update = False

    def run(self, node: ast.Module) -> ast.Module:
        """ Apply transformation and dependencies and fix new node location."""
        # TODO: Does it actually fixes the new node locations ? I don't think so.
        n = super().run(node)
        # the transformation updated the AST, so analyses may need to be rerun
        if self.update:
            self.passmanager._module_transformed(self, node)

        return n

    def apply(self, node):
        """ Apply transformation and return if an update happened. """
        new_node = self.run(node)
        return self.update, new_node


class _Project(object):...

Project = _Project()
"""
A sentinel object to indicate to gather() to run the analyses on
all modules contained in the pass manager.
"""

class RootModuleMapping(Mapping[ast.AST, ast.Module]):
    
    def __init__(self, dispatcher: EventDispatcher) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.add_event_listener(
            ModuleAddedEvent, self._module_added
        )
        dispatcher.add_event_listener(
            ModuleChangedEvent, self._module_changed
        )
        dispatcher.add_event_listener(
            ModuleRemovedEvent, self._module_removed
        )

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[ast.AST, ast.Module] = {}
    
    def _module_added(self, event:ModuleAddedEvent|ModuleChangedEvent):
        newmod = event.mod.node
        for node in ast.walk(newmod):
            self.__data[node] = newmod
    
    def _module_removed(self, event:ModuleRemovedEvent|ModuleChangedEvent):
        # O(n), every time :/
        node = event.mod.node
        to_remove = []
        for n,r in self.__data.items():
            if r is node:
                to_remove.append(n)
        for n in to_remove:
            del self.__data[n]
    
    def _module_changed(self, event:ModuleChangedEvent):
        self._module_removed(event)
        self._module_added(event)

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: ast.AST) -> ast.Module:
        return self.__data[__key]
    
    def __iter__(self) -> Iterator[ast.AST]:
        return iter(self.__data)
    
    def __len__(self) -> int:
        return len(self.__data)


class PassManager:
    '''
    Front end to the pass system.
    One pass manager can be used for the analysis of a collection of modules.
    '''

    def __init__(self):
        self._dispatcher = d = EventDispatcher()
        r = RootModuleMapping(d)
        
        self._cache = PassManagerCache(r)
        self.modules = ModuleCollection(d, r)
    
    def add_module(self, mod: Module):
        """
        Adds a new module to the pass manager.
        Use PassManager.modules to access modules.
        """
        # self.modules._add(mod)
        # alert mapping proxies that a module has been added
        self._dispatcher.dispatch_event(
            ModuleAddedEvent(
                mod
        ))
    
    def remove_module(self, mod:Module):
        """
        Remove a module from the passmanager. 
        This will allow adding another module with the same name or same module node.
        """
        # self.modules._remove(mod)
        self._dispatcher.dispatch_event(
            ModuleRemovedEvent(
                mod
        ))
    
    def _module_transformed(self, transformation: Transformation, node: ast.Module):
        self._dispatcher.dispatch_event(
            ModuleChangedEvent(self.modules[node]))
        
        # the transformation updated the AST, so analyses may need to be rerun
        # Instead of clearing the entire cache, only invalidate analysis that are affected
        # by the transformation.
        for analysis in self._cache.iter_analysis_types():
            if analysis not in transformation.preserves_analysis:
                # a transformation should only affects the analysis results in the current module. 
                # If a transformation is assumed not to change the externel interface of the module
                # we can safely keep all analyses that belong to other modules.
                self._invalidate(analysis, node)        

    def _invalidate(self, analysis: type[Analysis], node: ast.Module):
        """
        Clear the cache from this analysis.
        """
        # cases:
        # NodeAnalysis
        # FunctionAnalysis
        # ClassAnalysis
        # ModuleAnalysis
        # AnalysisAdaptor

        assert isinstance(node, ast.Module)
        
        # If the analysis type is adaptor, we don't bother clearing it
        # because it's result listens to the event and will update it's content.
        if not issubclass(analysis, AnalysisAdaptor):
            self._cache.clear(analysis, node)

        # alert mapping proxies that this analysis has been invalidated
        self._dispatcher.dispatch_event(
            InvalidatedAnalysisEvent(
                analysis, node
        ))

    # How to handle cycles ? With a context manager that will push onto a set of running analyses.
    @overload
    def gather(self, 
               analysis:type[Analysis[RunsOnT, ReturnsT]], 
               node:RunsOnT) -> ReturnsT:...
    @overload
    def gather(self, 
               analysis:type[ClassAnalysis[RunsOnT, ReturnsT]|FunctionAnalysis[RunsOnT, ReturnsT]], 
               node:ast.Module) -> Mapping[_FuncOrClassTypes, ReturnsT]:...
    @overload
    def gather(self, 
               analysis:type[ModuleAnalysis[RunsOnT, ReturnsT]], 
               node:_Project) -> Mapping[ast.Module, ReturnsT]:...
    @overload
    def gather(self, 
               analysis:type[ClassAnalysis[RunsOnT, ReturnsT]|FunctionAnalysis[RunsOnT, ReturnsT]], 
               node:_Project) -> ReturnsMapT:...
    def gather(self, 
               analysis:type[Analysis[RunsOnT, ReturnsT]], 
               node:RunsOnT|ast.Module|_Project) -> ReturnsT:
        """
        High-level function to call an ``analysis``.
        """
        # t0 = time()
        
        if not issubclass(analysis, (NodeAnalysis, FunctionAnalysis, 
                ClassAnalysis, ModuleAnalysis, AnalysisAdaptor)):
            raise TypeError(f'Wrong analysis type: {analysis}')
        
        # Promote the analysis if necessary
        if isinstance(node, ast.Module): 
            if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
                # scope to module promotions
                analysis = partialclass(ScopeToModuleAnalysis, analysis)
            # elif issubclass(analysis, NodeAnalysis):
            #     raise TypeError('NodeAnalysis cannot be promoted to module wide analysis')
        elif isinstance(node, _Project):
            if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
                # scope to project promotions
                analysis = partialclass(AllModulesAnalysis, 
                                        partialclass(ScopeToModuleAnalysis, 
                                                     analysis))
            elif issubclass(analysis, (ModuleAnalysis)):
                # module to project promotions
                analysis = partialclass(AllModulesAnalysis, analysis)
            elif issubclass(analysis, NodeAnalysis):
                raise TypeError('NodeAnalysis cannot be promoted to project wide analysis')

        a = analysis()
        a.attach(self)
        ret = a.run(node)
        
        # if run_times is not None: 
        #     run_times[analysis] = run_times.get(analysis, 0) + time()-t0
        
        return ret

    def apply(self, transformation, node):
        '''
        High-level function to call a `transformation' on a `node'.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        '''
        # t0 = time()
        assert issubclass(transformation, (Transformation, Analysis))
        a = transformation()
        a.attach(self)
        ret = a.apply(node)
        # if run_times is not None: run_times[transformation] = run_times.get(transformation,0) + time()-t0
        return ret


class ancestors(ModuleAnalysis[Mapping[ast.AST, list[ast.AST]]], ast.NodeVisitor):
    '''
    Associate each node with the list of its ancestors

    Based on the tree view of the AST: each node has the Module as parent.
    The result of this analysis is a dictionary with nodes as key,
    and list of nodes as values.

    >>> from pprint import pprint
    >>> mod = ast.parse('v = lambda x: x+1; w = 2')
    >>> pm = PassManager('t')
    >>> pprint({location(n):[location(p) for p in ps] for n,ps in pm.gather(ancestors, mod).items()})
    {'ast.Add at ?:?': ['ast.Module at ?:?',
                        'ast.Assign at ?:1',
                        'ast.Lambda at ?:1:4',
                        'ast.BinOp at ?:1:14'],
     'ast.Assign at ?:1': ['ast.Module at ?:?'],
     'ast.Assign at ?:1:19': ['ast.Module at ?:?'],
     'ast.BinOp at ?:1:14': ['ast.Module at ?:?',
                             'ast.Assign at ?:1',
                             'ast.Lambda at ?:1:4'],
     'ast.Constant at ?:1:16': ['ast.Module at ?:?',
                                'ast.Assign at ?:1',
                                'ast.Lambda at ?:1:4',
                                'ast.BinOp at ?:1:14'],
     'ast.Constant at ?:1:23': ['ast.Module at ?:?', 'ast.Assign at ?:1:19'],
     'ast.Lambda at ?:1:4': ['ast.Module at ?:?', 'ast.Assign at ?:1'],
     'ast.Load at ?:?': ['ast.Module at ?:?',
                         'ast.Assign at ?:1',
                         'ast.Lambda at ?:1:4',
                         'ast.BinOp at ?:1:14',
                         'ast.Name at ?:1:14'],
     'ast.Module at ?:?': [],
     'ast.Name at ?:1': ['ast.Module at ?:?', 'ast.Assign at ?:1'],
     'ast.Name at ?:1:11': ['ast.Module at ?:?',
                            'ast.Assign at ?:1',
                            'ast.Lambda at ?:1:4',
                            'ast.arguments at ?:?'],
     'ast.Name at ?:1:14': ['ast.Module at ?:?',
                            'ast.Assign at ?:1',
                            'ast.Lambda at ?:1:4',
                            'ast.BinOp at ?:1:14'],
     'ast.Name at ?:1:19': ['ast.Module at ?:?', 'ast.Assign at ?:1:19'],
     'ast.Param at ?:?': ['ast.Module at ?:?',
                          'ast.Assign at ?:1',
                          'ast.Lambda at ?:1:4',
                          'ast.arguments at ?:?',
                          'ast.Name at ?:1:11'],
     'ast.Store at ?:?': ['ast.Module at ?:?',
                          'ast.Assign at ?:1:19',
                          'ast.Name at ?:1:19'],
     'ast.arguments at ?:?': ['ast.Module at ?:?',
                              'ast.Assign at ?:1',
                              'ast.Lambda at ?:1:4']}
    '''

    def generic_visit(self, node):
        self.result[node] = current = self.current
        self.current += node,
        super().generic_visit(node)
        self.current = current

    visit = generic_visit

    def do_pass(self, node: ast.Module) -> Dict[ast.AST, Module]:
        self.result = dict()
        self.current = tuple()
        self.visit(node)
        return self.result

class root_module(ModuleAnalysis[Dict[ast.AST, Module]], ast.NodeVisitor):
    """
    Associate each node with it's root module.
    """       
    
    def generic_visit(self, node: ast.AST):
        self.result[node] = self.root_module
        super().generic_visit(node)
    
    def do_pass(self, node: Module) -> Dict[ast.AST, Module]:
        self.result = {}
        self.root_module = self.passmanager.modules[node]
        self.visit(node)
        return self.result
    
    visit = generic_visit

    


# modules = ModuleCollection()
# modules.add_module(Module(
#     ast.parse('v = lambda x: x+1; w = 2'), 
#     filename='test.py',
#     modname='test'))
# pm = PassManager(modules.get_module('test'))


# how to aggregate results of several analysis
# -> if analysis implements mapping we can use chainmaps
# how to initialize the builtins like we do in driver.py?


# pm = PassManager()
# pm.add_module(Module(ast.Module(body=[ast.FunctionDef()]), 'test', 'test.py'))
# pm.add_module(Module(ast.Module(), 'test2', 'test.py'))
# pm.add_module(Module(ast.Module(), 'test3', 'test.py'))

# d = pm.gather_for_all_modules(...)
# fn = ...
# d[fn]