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

def walk(node:ast.AST, typecheck:type|None=None, stopTypecheck:type|None=None):
    """
    Recursively yield all nodes matching the typecheck 
    in the tree starting at *node* (excluding *node* itself), in bfs order.

    Do not recurse on children of types matching the stopTypecheck type.

    See also `ast.walk` that behaves diferently.
    """
    from collections import deque
    todo = deque(ast.iter_child_nodes(node))
    while todo:
        node = todo.popleft()
        if stopTypecheck is None or not isinstance(node, stopTypecheck):
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

    def __init__(self) -> None:
        self._events: dict[type[Event], list[EventListener]] = {}
    
    # def update_from_other(self, other:EventDispatcher) -> None:
    #     """
    #     Update this dispatcher with the listeners another dispatcher.
    #     """
    #     for event_type, listeners in other._events.items():
    #         for l in listeners:
    #             self.addEventListener(event_type, l)

    def hasListener(self, event_type: type[Event], listener: EventListener) -> bool:
        """
        Return true if listener is register to event_type
        """
        # Check for event type and for the listener
        if event_type in self._events:
            return listener in self._events[event_type]
        else:
            return False

    def dispatchEvent(self, event: Event) -> None:
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

    def addEventListener(self, event_type: type[Event], listener: EventListener) -> None:
        """
        Add an event listener for an event type
        """
        # Add listener to the event type
        if not self.hasListener(event_type, listener):
            listeners = self._events.get(event_type, [])
            listeners.append(listener)
            self._events[event_type] = listeners

    def removeEventListener(self, event_type: type[Event], listener: EventListener) -> None:
        """
        Remove event listener.
        """
        # Remove the listener from the event type
        if self.hasListener(event_type, listener):
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

        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)
    
    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
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
    
    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
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

class CachedAnalysis(abc.ABC):
    @property
    @abc.abstractmethod
    def result(self) -> object:
        ...

    @classmethod
    def Success(self, result):
        return _CachedResult(result)
    
    @classmethod
    def Error(self, exception):
        return _CachedError(exception)

    @property
    def is_success(self):
        return isinstance(self, _CachedResult)

    @property
    def is_error(self):
        return isinstance(self, _CachedError)
    

@dataclasses.dataclass(frozen=True)
class _CachedError(CachedAnalysis):
    _error: Exception
    @property
    def result(self):
        raise self._error

@dataclasses.dataclass(frozen=True)
class _CachedResult(CachedAnalysis):
    _result: object
    @property
    def result(self):
        return self._result

class AnalysisCache:
    """
    The strucutre of the cache consist in nested dicts. 
    But this class facilitates the messages with the module pass manager.
    """

    def __init__(self) -> None:
        self.__data: dict[type[Analysis], dict[ast.AST, CachedAnalysis]] = defaultdict(dict)

    def set(self, analysis: type[Analysis], node: ast.AST, result: CachedAnalysis):
        """
        Store the analysis result in the cache.
        """        
        self.__data[analysis][node] = result
    
    def get(self, analysis: type[Analysis], node: ast.AST) -> Optional[CachedAnalysis]:
        """
        Query for the cached result of this analysis.
        """
        try:
            return self.__data[analysis][node]
        except KeyError:
            return None

    def clear(self, analysis: type[Analysis]):
        """
        Get rid of the the given analysis result.
        """
        if analysis in self.__data:
            del self.__data[analysis]
    
    def iter_analysis_types(self) -> Iterator[type[Analysis]]:
        yield from self.__data

T = TypeVar('T')
RunsOnT = TypeVar('RunsOnT')
ReturnsT = TypeVar('ReturnsT')
ChildAnalysisReturnsT = TypeVar('ChildAnalysisReturnsT')


class BasePass(Generic[RunsOnT, ReturnsT], abc.ABC):

    dependencies: Sequence[type[BasePass]] = ()
    """
    Statically declared dependencies
    """

    usesModules: bool = False
    """
    Whether the pass uses the passmanager.modules mapping;
    i.e. this analysis needs an iter-modules passmanager.

    If this flag is set to True, the self.passmanager attribute will
    be an instance of `ChildModulePassManager` that provides the 'modules' attribute.
    This kinf of analysis cannot be used with `ModulePassManager.Single`.
    """

    @classmethod
    def _usesModulesTransitive(cls) -> bool:
        return cls.usesModules or any(p._usesModulesTransitive() for p in cls.dependencies)

    def attach(self, pm: ModulePassManager):
        self.passmanager = pm

    def _verifyDependencies(self):
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
        self._verifyDependencies()
        
        if self._usesModulesTransitive() and not isinstance(self.passmanager, ChildModulePassManager):
            raise TypeError('This analysis uses other modules, you must use it through a PassManager instance.')

        for analysis in self.dependencies:
            if issubclass(analysis, Transformation):
                # Transformations always run the module since there 
                # are only module wide transformations at the moment.
                self.passmanager.apply(analysis, 
                                       self.passmanager.module.node)
            elif issubclass(analysis, Analysis):
                gather_on_node = node
                if issubclass(analysis, ModuleAnalysis):
                    gather_on_node = self.passmanager.module.node

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
        cached_result = self.passmanager.cache.get(typ, node)
        if cached_result is not None: # the result is cached
            result = cached_result.result # will rase an error if the initial analysis raised
        else:
            try:
                result = super().run(node)
            except Exception as e:
                self.passmanager.cache.set(typ, node, CachedAnalysis.Error(e))
                raise
            else:
                if not isinstance(self, AnalysisProxy):
                    # only set values in the cache for non-adaptor analyses.
                    self.passmanager.cache.set(typ, node, CachedAnalysis.Success(result))
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

class AnalysisProxy(
    Analysis[ast.Module, Mapping['_FuncOrClassTypes', ChildAnalysisReturnsT]], 
    Generic[ChildAnalysisReturnsT]):
    """
    A module analysis that returns a simple structure proxy for containing nodes.
    """

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
        return AnalysisProxyResult(self.passmanager, 
                node, self.__analysis_type, self.__keys_factory)


class AnalysisProxyResult(Mapping['_FuncOrClassTypes', ChildAnalysisReturnsT], 
                                  Generic[ChildAnalysisReturnsT]):
    # used for class/function to module analysis promotions

    def __init__(self, passmanager: ModulePassManager, 
                 module: ast.Module, 
                 analysis_type: type[Analysis[_FuncOrClassTypes, ChildAnalysisReturnsT]],
                 keys_factory: Callable,
                 ) -> None:
    
        assert issubclass(analysis_type, (FunctionAnalysis, ClassAnalysis))

        self.__module = module
        self.__analysis_type = analysis_type
        self.__keys_factory = keys_factory
        self.__keys = keys_factory(module)
        # register the event listener
        passmanager._dispatcher.addEventListener(
            ModuleChangedEvent, self._onModuleChangedEvent
        )
        self.__pm = passmanager
    
    def keys(self) -> Collection[_FuncOrClassTypes]: # type:ignore[override]
        return self.__keys
    
    def __contains__(self, __key: object) -> bool:
        return __key in self.__keys

    def __getitem__(self, __key: _FuncOrClassTypes) -> ChildAnalysisReturnsT:
        if __key not in self.__keys:
            raise KeyError(__key)
        return self.__pm.gather(self.__analysis_type, __key)
    
    def __iter__(self) -> Iterator[_FuncOrClassTypes]:
        return iter(self.__keys)
    
    def __len__(self) -> int:
        return len(self.__keys)

    def _onModuleChangedEvent(self, event: ModuleChangedEvent) -> None:
        # if the node is the module.
        node = event.mod.node
        if node is self.__module:
            # refresh keys
            self.__keys = self.__keys_factory(self.__module)

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

    def run(self, node: ast.Module=None) -> ast.Module:
        """ Apply transformation and dependencies and fix new node location."""
        if node is None:
            node = self.passmanager.module.node
        else:
            assert node is self.passmanager.module.node
        
        # TODO: Does it actually fixes the new node locations ? I don't think so.
        n = super().run(node)
        # the transformation updated the AST, so analyses may need to be rerun
        if self.update:
            self.passmanager._moduleTransformed(self, self.passmanager.module)

        return n

    def apply(self, node):
        """ Apply transformation and return if an update happened. """
        new_node = self.run(node)
        return self.update, new_node


class RootModuleMapping(Mapping[ast.AST, ast.Module]):
    
    def __init__(self, dispatcher: EventDispatcher) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(
            ModuleAddedEvent, self._onModuleAddedEvent
        )
        dispatcher.addEventListener(
            ModuleChangedEvent, self._onModuleChangedEvent
        )
        dispatcher.addEventListener(
            ModuleRemovedEvent, self._onModuleRemovedEvent
        )

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[ast.AST, ast.Module] = {}
    
    def _onModuleAddedEvent(self, event:ModuleAddedEvent|ModuleChangedEvent) -> None:
        newmod = event.mod.node
        for node in ast.walk(newmod):
            self.__data[node] = newmod
    
    def _onModuleRemovedEvent(self, event:ModuleRemovedEvent|ModuleChangedEvent) -> None:
        # O(n), every time :/
        node = event.mod.node
        to_remove = []
        for n,r in self.__data.items():
            if r is node:
                to_remove.append(n)
        for n in to_remove:
            del self.__data[n]
    
    def _onModuleChangedEvent(self, event:ModuleChangedEvent) -> None:
        self._onModuleRemovedEvent(event)
        self._onModuleAddedEvent(event)

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data
    def __getitem__(self, __key: ast.AST) -> ast.Module:
        return self.__data[__key]
    def __iter__(self) -> Iterator[ast.AST]:
        return iter(self.__data)
    def __len__(self) -> int:
        return len(self.__data)


class PassManagerBase(abc.ABC):
    
    @abc.abstractmethod
    def gather(self, analysis, node):...
    
    @abc.abstractmethod
    def apply(self, transformation, node):...


class ModulePassManager(PassManagerBase):
    '''
    Front end to the module-level pass system.
    One `ModulePassManager` can only be used to analyse one module.
    '''

    def __init__(self, module: Module, dispatcher:EventDispatcher) -> None:
        """
        Private method.

        :param module: 
        :param dispatcher: `PassManager`'s dispatcher or None for single module usage.

        """
        self.module = module
        self.cache = AnalysisCache()
        self._dispatcher = dispatcher
        self._dispatcher.addEventListener(InvalidatedAnalysisEvent, 
                                          self._onInvalidatedAnalysisEvent)

    @classmethod
    def Single(cls, node:ast.Module, 
                 modname: str = '',
                 filename: str | None = None,
                 is_package: bool = False,
                 is_namespace_package: bool = False,
                 is_stub: bool = False,
                 code: str | None = None,) -> ModulePassManager:
        """
        Create a standalone module pass manager.
        
        >>> mpm = ModulePassManager.Single(ast.parse('pass'), 'test')
        >>> assert isinstance(mpm.module.node, ast.Module)

        :param modname:
        :param filename:
        :param is_package:
        :param is_namespace_package:
        :param is_stub:
        :param code:
        """
        return cls(Module(node, modname, filename, 
                          is_package=is_package, 
                          is_namespace_package=is_namespace_package, 
                          is_stub=is_stub, 
                          code=code), 
                          # A standalone dispatcher for this module.
                          EventDispatcher())

    def gather(self, analysis, node=None):
        if not issubclass(analysis, Analysis):
            raise TypeError(f'unexpected analysis type: {analysis}')
        
        if node is None:
            node = self.module.node
        
        # Promote the analysis if necessary
        if isinstance(node, ast.Module): 
            if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
                # scope to module promotions
                analysis = partialclass(AnalysisProxy, analysis)

        a = analysis()
        a.attach(self)
        ret = a.run(node)
        return ret
    
    def apply(self, transformation, node=None):
        if not issubclass(transformation, (Transformation, Analysis)):
            raise TypeError(f'unexpected analysis type: {transformation}')
        
        if node is None:
            node = self.module.node

        if not isinstance(node, ast.Module):
            raise TypeError(f'unexpected node type: {node}')

        a = transformation()
        a.attach(self)
        ret = a.apply(node)
        # if run_times is not None: run_times[transformation] = run_times.get(transformation,0) + time()-t0
        return ret
    
    def _iterPassManagers(self) -> Iterable[ModulePassManager]:
        return (self, )

    def _moduleTransformed(self, transformation: Transformation, mod: Module):
        """
        Alert that the given module has been transformed, this is automatically called 
        at the end of a triaformation if it updated the module.
        """
        self._dispatcher.dispatchEvent( # this is for the root modules mapping.
            ModuleChangedEvent(mod))
        
        # the transformation updated the AST, so analyses may need to be rerun
        # Instead of clearing the entire cache, only invalidate analysis that are affected
        # by the transformation.
        invalidated_analyses:set[type[Analysis]] = ordered_set()
        for mpm in self._iterPassManagers():
            for analysis in mpm.cache.iter_analysis_types():
                if (
                    # if the analysis is explicitely presedved by this transform,
                    # do not invalidate.
                    (analysis not in transformation.preserves_analysis) 
                    and ( 
                        # if it's not explicately preserved and the transform affects the module
                        # invalidate.             or if the analysis requires other modules
                        (mpm.module.node is mod.node) or (analysis._usesModulesTransitive()))):
                    
                    invalidated_analyses.add(analysis)
        
        for analys in invalidated_analyses:
            # alert that this analysis has been invalidated
            self._dispatcher.dispatchEvent(
                InvalidatedAnalysisEvent(
                    analys, mod.node
            ))

    def _onInvalidatedAnalysisEvent(self, event: InvalidatedAnalysisEvent):
        """
        Clear the cache from this analysis.
        """
        # cases:
        # NodeAnalysis
        # FunctionAnalysis
        # ClassAnalysis
        # ModuleAnalysis
        # AnalysisAdaptor -> cannot be

        analysis: type[Analysis] = event.analysis
        node: ast.Module = event.node

        assert isinstance(node, ast.Module)
        # It cannot be an adaptor because we don't store them in the cache
        assert not issubclass(analysis, AnalysisProxy)
        
        if analysis._usesModulesTransitive() or node is self.module.node:
            self.cache.clear(analysis)
        
class ChildModulePassManager(ModulePassManager):

    def __init__(self, module: Module, passmanager: PassManager) -> None:
        super().__init__(module, passmanager._dispatcher)
        self._passmanager = passmanager

    @property
    def modules(self):
        """
        Proxy to other modules.
        """
        return self._passmanager.modules

    def _iterPassManagers(self) -> Iterable[ModulePassManager]:
        return self._passmanager._passmanagers.values()

    def gather(self, analysis, node=None):
        if node is None or self.modules[node] is self.module:
            return super().gather(analysis, node)
        return self._passmanager.gather(analysis, node)
    
    def apply(self, transformation, node=None):
        if node is None or self.modules[node] is self.module:
            return super().apply(transformation, node)
        return self._passmanager.apply(transformation, node)

class ChildPassManagers(Mapping[Module, ChildModulePassManager]):
    def __init__(self, passmanager:PassManager) -> None:
        super().__init__()
        self._passmanager = passmanager

        # register the event listeners
        passmanager._dispatcher.addEventListener(
            ModuleAddedEvent, self._onModuleAddedEvent
        )
        passmanager._dispatcher.addEventListener(
            ModuleRemovedEvent, self._onModuleRemovedEvent
        )

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[Module, ChildModulePassManager] = {}

    def _onModuleAddedEvent(self, event:ModuleAddedEvent) -> None:
        self.__data[event.mod] = ChildModulePassManager(event.mod, self._passmanager)
    
    def _onModuleRemovedEvent(self, event:ModuleRemovedEvent) -> None:
        del self.__data[event.mod]

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data
    def __getitem__(self, __key: Module) -> ChildModulePassManager:
        return self.__data[__key]
    def __iter__(self) -> Iterator[Module]:
        return iter(self.__data)
    def __len__(self) -> int:
        return len(self.__data)
    
class PassManager(PassManagerBase):
    '''
    Front end to the inter-modules pass system.
    One `PassManager` can be used for the analysis of a collection of modules.
    '''

    def __init__(self):
        d = EventDispatcher()
        r = RootModuleMapping(d)
        self.modules: Mapping[str | ast.AST, Module] = ModuleCollection(d, r)
        self._dispatcher = d
        self._passmanagers: Mapping[Module, ChildModulePassManager] = ChildPassManagers(self)
        
    
    def add_module(self, mod: Module):
        """
        Adds a new module to the pass manager.
        Use PassManager.modules to access modules.
        """
        self._dispatcher.dispatchEvent(
            ModuleAddedEvent(
                mod
        ))
    
    def remove_module(self, mod:Module):
        """
        Remove a module from the passmanager. 
        This will allow adding another module with the same name or same module node.
        """
        self._dispatcher.dispatchEvent(
            ModuleRemovedEvent(
                mod
        ))
    
    # How to handle cycles ? With a context manager that will push onto a set of running analyses.
    @overload
    def gather(self, 
               analysis:type[Analysis[RunsOnT, ReturnsT]], 
               node:RunsOnT) -> ReturnsT:...
    @overload
    def gather(self, 
               analysis:type[ClassAnalysis[ReturnsT]|FunctionAnalysis[ReturnsT]], 
               node:ast.Module) -> Mapping[_FuncOrClassTypes, ReturnsT]:...
    def gather(self, 
               analysis:type[Analysis[RunsOnT, ReturnsT]], 
               node:RunsOnT|ast.Module) -> ReturnsT | Mapping[_FuncOrClassTypes, ReturnsT]:
        """
        High-level function to call an ``analysis`` on any node in the system.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.gather(analysis, node)

    def apply(self, transformation, node):
        '''
        High-level function to call a `transformation' on a `node'.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        '''
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.apply(transformation, node)

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