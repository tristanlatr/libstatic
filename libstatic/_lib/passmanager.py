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

# 1 - Only true analysis results should be in there # DONE
# adapters do not belong in the cache since they can be re-created on the fly # DONE

# 2 - The data should only be cached at one place:
# ScopeToModuleAnalysisResult should not store the data but always delegate to gather: # DONE
# this way we don't have to register the mapping proxy as listener to the invalidated analysis event
# since it's not storing any data.

#  -> There is no such things as a analysis that runs on all modules # DONE
#       This will be called client objects/façade and we don't care about them here: 
#       these objects are composed by the PassManager and redirect
#       all requests. 

# - PassManager should rely on lower level ModulePassManager that handles everything # DONE
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

# TODO: Work for having function and classes transformations 
# that will invalidate only the required function analyses.
# Seulement les transformations de modules peuvent changer l'interface
# publique d'un module (ex changer signature d'une fonction ou autre).
# Une transformation d'un certain niveau ne doit pas interferrer avec 
# les aures transformation du même niveau.
# Exemple: Pas possible d'ajouter des global ou nonlocal keyword depuis 
# une transformation de fonction.
# Compatibilité avec NodeTransformer + NodeVisitor pour les analyses.
# Comment determiner si la pass update l'ast: calculer un hash avant et apres. 

# Ressources: 
# LLVM PassManager documentation: https://llvm.org/doxygen/classllvm_1_1PassManager.html
from __future__ import annotations

from collections import defaultdict
import enum
from functools import lru_cache, partialmethod, partial
import itertools
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Hashable, Iterable, Iterator, Mapping, Optional, Self, Tuple, TypeVar, Generic, Sequence, Type, Union, TypeAlias, final, overload
import ast
from time import time
from contextlib import contextmanager
from pathlib import Path
import dataclasses
import abc

from beniget.ordered_set import ordered_set

from .exceptions import NodeLocation

@lru_cache(maxsize=None, typed=True)
def newsubclass(cls: type, **class_dict: Hashable) -> type:
    """
    Create a subclass with the same caracteristics as 
    the given class but add the provided class attributes.
    """
    # DO NOT create a subclass if all class_dict items are already set to their values.
    if all(getattr(cls, k, object()) == v for k,v in class_dict.items()):
        return cls
    newcls = type(cls.__name__, (cls, ), class_dict)
    assert newcls.__name__ == cls.__name__
    newcls.__qualname__ = cls.__qualname__
    assert isinstance(newcls, type)
    return newcls


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

class PassMeta(abc.ABCMeta):

    # The mata type is only necessary to implement the subclassing with arguments
    # feature of the Pass class. 
    # We should not add methods to the meta type unless absolutely required.
    # This hack limits the subclass explosition issue (https://python-patterns.guide/gang-of-four/composition-over-inheritance/)
    # by creating cached subclasses when calling the class with arguments.

    # TODO: required parameter should be supported as positional arguments.
    # but we need to handle when there are several values for one argument.
    def __call__(cls: Pass, **kwargs: Hashable) -> Any:
        if not kwargs:
            # create instance only when calling with no arguments.
            return super().__call__()
        
        # other wise created a derived type that binds the class attributes. 
        # This is how we create analysis with several options.
        return cls._newTypeWithOptions(**kwargs)

class PassDependencyDescriptor:
    def __init__(self, callback:Callable[[], Any]) -> None:
        self.callback = callback

class Pass(Generic[RunsOnT, ReturnsT], abc.ABC, metaclass=PassMeta):

    dependencies: Sequence[type[Pass]] = ()
    """
    Statically declared dependencies
    """

    requiredParameters: tuple[str] = ()
    """
    Some passes looks for specific name that needs to be provided as arguments. 
    """
    
    optionalParameters: dict[str, Any] = {}
    """
    Other optional arguments to their default values.
    """

    passmanager: ModulePassManager
    
    @classmethod
    def _newTypeWithOptions(cls, **kwargs):
        # verify no junk arguments are slipping throught.
        validNames = ordered_set((*cls.requiredParameters, *cls.optionalParameters))
        try:
            junk = next(p for p in kwargs if p not in validNames)
        except StopIteration:
            pass
        else:
            raise TypeError(f'{cls.__qualname__}() does not recognize keyword {junk!r}')
        
        # remove the arguments that are already set to their values.
        kwargs = {k:v for k,v in kwargs.items() if getattr(cls, k, object()) != v}
        
        # if any of the arguments are not set to thei default values (or if a required argument is already set), 
        # it means we're facing a case of doudle calling the class: this is not supported and we raise an error.
        # it's supported because it might creates two different classes that do exactly the same thing :/
        _nothing = object()
        if any(getattr(cls, promlematic:=a, _nothing) is not _nothing for a in cls.requiredParameters) or \
            any(getattr(cls, promlematic:=a, _d) is not _d for a,_d in cls.optionalParameters.items()):
            hint = ''
            if cls.__qualname__ != cls.__bases__[0].__qualname__:
                hint = ' (hint: analysis subclassing is not supported)'

            raise TypeError(f"Specifying parameter {promlematic!r} this way is not supported{hint}, "
                            f"you must list all parameters into a single call to class {cls.__bases__[0].__qualname__}")
        
        # This will automatically trigger __init_subclass__, but just once because the resulting class is cached.
        return newsubclass(cls, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Set the default values for optional arguments
        _nothing = object()
        for p, default in cls.optionalParameters.items():
            if getattr(cls, p, _nothing) is _nothing:
                setattr(cls, p, default)
        
        cls.prepareClass()

    @classmethod
    def prepareClass(cls):
        """
        Here you can dynamically adjust dependencie based on arguments.
        Call super().prepareClass(cls) after adjusting the dependancies.

        This is called from the __init_subclass__ hook.
        """
        cls._verifyDepNameClash()
    
    @classmethod
    def _verifyDepNameClash(cls):

        # verify nothing conflicts with the dependencies names
        _nothing = object()
        try:
            clash = next(p for p in (d.__name__ for d in cls.dependencies) if getattr(cls, p, _nothing) is not _nothing)
        except StopIteration:
            pass
        else:
            raise TypeError(f'{cls.__qualname__}: invalid class declaration, name {clash!r} is already taken by a dependency')

    @classmethod
    def _getAllDependencies(cls, _seen:set[type[Analysis]]=None) -> Collection[type[Pass]]:
        seen = _seen or ordered_set()
        for d in cls.dependencies:
            if d in seen:
                continue
            seen.add(d)
            d._getAllDependencies(seen)
        return None if _seen else seen
    
    @classmethod
    def _isAbstract(cls) -> bool:
        try:
            cls._verifyRequiredParamaters()
        except TypeError:
            return True
        if cls.do_pass is Pass.do_pass:
            return True
        return False

    @classmethod
    def _usesModulesTransitive(cls) -> bool:
        return any(d is modules for d in cls._getAllDependencies())

    def __getattribute__(self, name):
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, PassDependencyDescriptor):
            return attr.callback()
        return attr

    def _verifyDependencies(self):
        """
        """
        # Any kinf of analsis can depend on any other.
         
    @classmethod
    def _verifyRequiredParamaters(cls):
        # verify required arguments exists
        _nothing = object()
        try:
            missing = next(p for p in cls.requiredParameters if getattr(cls, p, _nothing) is _nothing)
        except StopIteration:
            pass
        else:
            raise TypeError(f'{cls.__qualname__}() is missing keyword {missing!r}')
    
    def _verifyScoping(self):
        # verify scoping
        if self._usesModulesTransitive() and not isinstance(self.passmanager, ModulePassManager):
            raise TypeError('This analysis uses other modules, you must use it through a PassManager instance.')

    def prepare(self, node: RunsOnT):
        '''Gather analysis result required by this analysis'''
        self._verifyDependencies()
        self._verifyScoping()
        self._verifyRequiredParamaters()

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

                # Use a descriptors for all non-transformations.
                callback = partial(self.passmanager.gather, analysis, gather_on_node) # type:ignore
                setattr(self, analysis.__name__, PassDependencyDescriptor(callback))
            else:
                raise TypeError(f"dependencies should be a Transformation or an Analysis, not {analysis}")

    def attach(self, pm: ModulePassManager):
        self.passmanager = pm

    def run(self, node: RunsOnT) -> ReturnsT:
        """Override this to add special pre or post processing handlers."""
        self.prepare(node)
        return self.do_pass(node)
    
    @abc.abstractmethod
    def do_pass(self, node: RunsOnT) -> ReturnsT:
        """
        Override this to add actual pass logic. 
        """


class Analysis(Pass[RunsOnT, ReturnsT]):
    """
    A pass that does not change its content but gathers informations about it.
    """
    update = False # TODO: Is this needed?

    def run(self, node: RunsOnT) -> ReturnsT:
        typ = type(self)
        cached_result = self.passmanager.cache.get(typ, node)
        if cached_result is not None: # the result is cached
            result = cached_result.result # will rase an error if the initial analysis raised
        else:
            try:
                # this will call prepare().
                result = super().run(node)
            except Exception as e:
                self.passmanager.cache.set(typ, node, CachedAnalysis.Error(e))
                raise
            else:
                if not isinstance(self, UncachableAnalysis):
                    # only set values in the cache for non-proxy analyses.
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


class UncachableAnalysis(Analysis[RunsOnT, ReturnsT]):
    ...


class AnalysisProxy(
    UncachableAnalysis[ast.Module, Mapping['_FuncOrClassTypes', ChildAnalysisReturnsT]], 
    Generic[ChildAnalysisReturnsT]):
    """
    A module analysis that returns a simple structure proxy for containing nodes.
    """

    # the results of each analysis and make them accessible in the result dict proxy.
    # the result must watch invalidations in order to reflect the changes
    
    requiredParameters = ('analysis', )

    def __keys_factory(self, node: ast.Module) -> Collection[_FuncOrClassTypes]:
        if isinstance(self.analysis, ClassAnalysis):
            typecheck = (ast.ClassDef,)
        elif isinstance(self.analysis, FunctionAnalysis):
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
                node, self.analysis, self.__keys_factory)


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


class modules(UncachableAnalysis[None, ModuleCollection]):
    """
    This analysis returns the mapping of modules: L{ModuleCollection}.
    """
    
    passmanager: ModulePassManager
    
    def run(self, _: None) -> Any:
        # skip prepare() since this analysis is trivially special
        return self.do_pass(None)
    
    def do_pass(self, _: None) -> ModuleCollection:
        return self.passmanager._getModules(self)


class Transformation(Pass[ast.Module, ast.Module]):
    """A pass that updates its content."""
    
    preserves_analysis = ()
    """
    This variable should be overridden by subclasses to provide specific list.
    By default, assume all analyses are affected. 
    """

    update = False

    def run(self, node: ast.Module|None=None) -> ast.Module:
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
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleChangedEvent, self._onModuleChangedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

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


class ModulePassManager:
    '''
    Front end to the module-level pass system.
    One `ModulePassManager` can only be used to analyse one module.
    '''

    def __init__(self, module: Module, passmanager: ProjectPassManager) -> None:
        """
        """
        self.module = module
        self.cache = AnalysisCache()
        self.__pm = passmanager
        self.__pm._dispatcher.addEventListener(InvalidatedAnalysisEvent, 
                                          self._onInvalidatedAnalysisEvent)

    def gather(self, analysis: type[Analysis], node: ast.AST|None = None):
        
        if not issubclass(analysis, Analysis):
            raise TypeError(f'unexpected analysis type: {analysis}')
        
        if self.__pm.modules[node] is not self.module:
            return self.__pm.gather(analysis, node)

        if node is None:
            node = self.module.node
        
        # Promote the analysis if necessary
        if isinstance(node, ast.Module): 
            if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
                # scope to module promotions
                analysis = AnalysisProxy(analysis=analysis)
                assert issubclass(analysis, Analysis)

        a = analysis()
        a.attach(self)
        ret = a.run(node)
        return ret
    
    def apply(self, transformation: type[Pass], node: ast.Module|None = None):
        
        if not issubclass(transformation, (Transformation, Analysis)):
            raise TypeError(f'unexpected analysis type: {transformation}')
        
        if self.__pm.modules[node] is not self.module:
            return self.__pm.apply(transformation, node)

        if node is None:
            node = self.module.node

        if not isinstance(node, ast.Module):
            raise TypeError(f'unexpected node type: {node}')

        a = transformation()
        a.attach(self)
        ret = a.apply(node)
        return ret
    
    def _iterPassManagers(self) -> Iterable[ModulePassManager]:
        return self.__pm._passmanagers.values()

    def _moduleTransformed(self, transformation: Transformation, mod: Module):
        """
        Alert that the given module has been transformed, this is automatically called 
        at the end of a triaformation if it updated the module.
        """
        self.__pm._dispatcher.dispatchEvent( # this is for the root modules mapping.
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
            self.__pm._dispatcher.dispatchEvent(
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
        assert not issubclass(analysis, UncachableAnalysis)
        
        if analysis._usesModulesTransitive() or node is self.module.node:
            self.cache.clear(analysis)

    def _getModules(self, analysis:Analysis) -> ModuleCollection:
        if not isinstance(analysis, modules):
            raise RuntimeError(f'Only the analysis {modules.__qualname__!r} can access the ModuleCollection, please use that.')
        return self.__pm.modules

class PassManagerCollection(Mapping[Module, ModulePassManager]):
    
    def __init__(self, passmanager:ProjectPassManager) -> None:
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
        self.__data: dict[Module, ModulePassManager] = {}

    def _onModuleAddedEvent(self, event:ModuleAddedEvent) -> None:
        self.__data[event.mod] = ModulePassManager(event.mod, self._passmanager)
    
    def _onModuleRemovedEvent(self, event:ModuleRemovedEvent) -> None:
        del self.__data[event.mod]

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data
    def __getitem__(self, __key: Module) -> ModulePassManager:
        return self.__data[__key]
    def __iter__(self) -> Iterator[Module]:
        return iter(self.__data)
    def __len__(self) -> int:
        return len(self.__data)


class ProjectPassManager:
    '''
    Front end to the inter-modules pass system.
    One `PassManager` can be used for the analysis of a collection of modules.
    '''

    def __init__(self):
        d = EventDispatcher()
        r = RootModuleMapping(d)
        
        self.modules = ModuleCollection(d, r)
        
        self._dispatcher = d
        self._passmanagers = PassManagerCollection(self)
    
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


PassManager = ProjectPassManager

# modules = ModuleCollection()±±
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