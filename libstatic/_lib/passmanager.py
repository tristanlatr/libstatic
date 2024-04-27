"""
This module provides a framework for L{ast} pass management. 
It's a building block to write static analyzer or compiler for Python. 

There are two kinds of passes: transformations and analysis.
    * L{ModuleAnalysis}, L{FunctionAnalysis}, L{ClassAnalysis} and L{NodeAnalysis} are to be
      subclassed by any pass that collects information about the AST.
    * L{PassManager.gather} is used to gather (!) the result of an analyses on an AST node.
    * L{Transformation} is to be sub-classed by any pass that updates the AST.
    * L{PassManager.apply} is used to apply (sic) a transformation on an AST node.

To write an analysis: 

    - Subclass one of the analysis class cited above.
    - List analysis required by yours in the L{Analysis.dependencies} tuple, 
      they will be built automatically and stored in the attribute with the corresponding name.
    - Write your analysis logic inside the L{Analysis.doPass} method. The analysis result must be returned by
      the L{Analysis.doPass} method or an exeception raised.
    - Use it either from another pass’s C{dependencies}, or through the L{PassManager.gather} function.

B{Example}: In the following code snippets we'll look at how to write a type inference function that
understand literal values as well as builtins calls like L{list()} with a relatively high level of confidence
we're not misinterpreting a symbol name.   

Start by coding the logic of your lower-level analysis, which in our case computes the locals for a given scope:

>>> class CollectLocals(ast.NodeVisitor):
...    "Compute the set of identifiers local to a given node."
...    def __init__(self):
...        self.Locals = set()
...        self.NonLocals = set()
...    def visit_FunctionDef(self, node):
...        # no recursion
...        self.Locals.add(node.name)
...    visit_AsyncFunctionDef = visit_FunctionDef
...    visit_ClassDef = visit_FunctionDef
...    def visit_Nonlocal(self, node):
...        self.NonLocals.update(name for name in node.names)
...    visit_Global = visit_Nonlocal
...    def visit_Name(self, node):
...        if isinstance(node.ctx, ast.Store) and node.id not in self.NonLocals:
...            self.Locals.add(node.id)
...    def visit_arg(self, node):
...        self.Locals.add(node.arg)
...    def skip(self, _):
...        pass
...    visit_SetComp = visit_DictComp = visit_ListComp = skip
...    visit_GeneratorExp = skip
...    visit_Lambda = skip
...    def visit_Import(self, node):
...        for alias in node.names:
...            base = alias.name.split(".", 1)[0]
...            self.Locals.add(alias.asname or base)
...    def visit_ImportFrom(self, node):
...        for alias in node.names:
...            self.Locals.add(alias.asname or alias.name)

Then wraps it inside an Analysis subclass:

>>> class node_locals(NodeAnalysis[set[str]]):
...     def doPass(self, node):
...         visitor = CollectLocals()
...         visitor.generic_visit(node)
...         return visitor.Locals

Now let's try out our analysis, here we are hard coding the source code for testing purposes
but in real life you can use the L{Finder}. 

>>> src = '''
... import sys, os, set
... from twisted.python import reflect
... import twisted.python.filepath
... from twisted.python.components import registerAdapter
... foo, foo2 = 123, int()
... def doFoo():
...   import datetime
... class Foo:
...   x = 0
...   list = lambda x: True
...   y = list() or set()
...   def __init__(self):
...     self.a = [a for a in list()]
... _foo = "boring internal details"
... '''

This is where is gets interesting, how to actually use the framework

>>> pm = PassManager() # Create a PassManager instance
>>> pm.add_module(Module(ast.parse(src), 'testmodulename')) # Add your module(s) to it

When you add a module it's recorded in the L{PassManager.modules} mapping. It's a special mapping
that allows accessing values by module name or by any node contained in the module tree.

>>> testnode = pm.modules['testmodulename'].node
>>> testclass = next(n for n in testnode.body if isinstance(n, ast.ClassDef) and n.name == 'Foo')
>>> testmethod = next(n for n in testclass.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')

Gather analysis result with L{PassManager.gather}.

>>> print(sorted(pm.gather(node_locals, testnode) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted']

>>> print(sorted(pm.gather(node_locals, testclass) ))
['__init__', 'list', 'x', 'y']

>>> print(sorted(pm.gather(node_locals, testmethod) ))
['self']

Now let's not forget about our goal: builtins calls inference. So in order to be relatively sure we're not 
misinterpreting builtins names, we nee to know all accessible names from any expression nodes. If the name 'list' is not
declared it means it's an actual builtins call. 

Let's use one of the provided analysis: L{analyses.node_enclosing_scope} as another dependency.

>>> from . import analyses

Note how the analyses can both be used as instance attribute, in which case they are run on the current node; 
or as C{self.passmanager.gather()} argument to run them on any other nodes.

>>> ast_scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, 
...     ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Lambda, ast.Module)
>>> class declared_names(NodeAnalysis[set[str]]):
...     "Collects all the explicitely declared names accessible from the current node"
...     dependencies = (node_locals, analyses.node_enclosing_scope, )
...     def doPass(self, node):
...         declared = set(); i = 0; 
...         scope = self.node_enclosing_scope if not isinstance(node, ast_scopes) else node
...         while scope is not None:                      # first iteration. 
...             if not isinstance(scope, ast.ClassDef) or i==0:
...                 declared.update(self.passmanager.gather(node_locals, scope))
...             scope = self.passmanager.gather(analyses.node_enclosing_scope, scope)
...             i += 1
...         return declared

Let's try out this new analysis.

>>> print(sorted(pm.gather(declared_names, testnode) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted']

>>> print(sorted(pm.gather(declared_names, testclass) ))
['Foo', '__init__', '_foo', 'doFoo', 'foo', 'foo2', 'list', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted', 'x', 'y']

>>> print(sorted(pm.gather(declared_names, testmethod) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'self', 'set', 'sys', 'twisted']

Looks good... Now putting everything together:

>>> callfuncs2type = {'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'str': str, 'frozenset': frozenset, 'int': int}
>>> class infer_type(NodeAnalysis[type]):
...     '''
...     This class tries to infer the types of literal expression as 
...     well builtins calls like dict(). 
...     '''
...     dependencies = (declared_names, )
...     def doPass(self, node) -> type | None:
...         if (isinstance(node, ast.Call) and 
...             isinstance(node.func, ast.Name) and 
...             node.func.id in callfuncs2type):
...             if node.func.id not in self.declared_names:
...                 return callfuncs2type[node.func.id]
...             return None
...         try:
...             # try as literal
...             return type(ast.literal_eval(node))
...         except Exception:
...             return None

>>> testintcall = next(n for n in ast.walk(testnode) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'int')
>>> testsetcall = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'set')
>>> testlistcall1 = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testlistcall2 = next(n for n in ast.walk(testmethod) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testliteralstring = next(n for n in ast.walk(testnode) if isinstance(n, ast.Constant) and isinstance(n.value, str))

We now have a manner to statically infer types of builtins calls with relatively high confidence (It could be improved, this is just an example after all).

>>> pm.gather(infer_type, testintcall)
<class 'int'>
>>> pm.gather(infer_type, testsetcall) is None # fails because 'set' is an imported name
True
>>> pm.gather(infer_type, testlistcall1) is None # fails because 'list' is defined in class body
True
>>> pm.gather(infer_type, testlistcall2)
<class 'list'>
>>> pm.gather(infer_type, testliteralstring)
<class 'str'>

Sometime it is required for an analysis to have different behaviour depending on optional or required parameters.
To write an analysis taking parameters: 
    
    - List all required parameters names in the C{requiredParameters} tuple.
    - List all optional parameters in the C{optionalParameters} dict; 
      keys are parameters names are values are the default values in case none is provided.
    - To create your parameterized analysis simply call your analysis class with keywords coresponding to your
      parameters. This will create a devired subclass with the parameters set at class level.
    
To write a transformation...


"""

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

# If an analysis raises an exception during processing, this execption should be saved in the cache so it's re-raised # DONE
# whenever the same analysis is requested again.

# When a module is added or removed all transitively using modules analysis should be cleared from the cache. # DONE

# TODO: Integrate with a simple logging framework.

# TODO: Think of where to use weakrefs. 

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
# https://llvm.org/docs/WritingAnLLVMPass.html

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache, partial
from itertools import chain
import queue
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
    Generic,
    Sequence,
    Union,
    overload,
)
import ast
from pathlib import Path
import dataclasses

from beniget.ordered_set import ordered_set

@lru_cache(maxsize=None, typed=True)
def _newsubclass(cls: type, **class_dict: Hashable) -> type:
    """
    Create a subclass with the same caracteristics as
    the given class but add the provided class attributes.
    """
    # DO NOT create a subclass if all class_dict items are already set to their values.
    if all(getattr(cls, k, object()) == v for k, v in class_dict.items()):
        return cls
    newcls = type(cls.__name__, (cls,), class_dict)
    assert newcls.__name__ == cls.__name__
    newcls.__qualname__ = cls.__qualname__
    assert isinstance(newcls, type)
    return newcls


if TYPE_CHECKING:
    _FuncOrClassTypes = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    _ScopesTypes = (
        _FuncOrClassTypes
        | ast.SetComp
        | ast.DictComp
        | ast.ListComp
        | ast.GeneratorExp
        | ast.Lambda
    )
    _CannotContainClassOrFunctionTypes = (
        ast.expr
        | ast.Return
        | ast.Delete
        | ast.Assign
        | ast.AugAssign
        | ast.AnnAssign
        | ast.Raise
        | ast.Assert
        | ast.Import
        | ast.ImportFrom
        | ast.Global
        | ast.Nonlocal
        | ast.Expr
        | ast.Pass
        | ast.Break
        | ast.Continue
    )

else:
    _FuncOrClassTypes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    _ScopesTypes = (
        *_FuncOrClassTypes,
        ast.SetComp,
        ast.DictComp,
        ast.ListComp,
        ast.GeneratorExp,
        ast.Lambda,
    )
    _CannotContainClassOrFunctionTypes = (
        ast.expr,
        ast.Return,
        ast.Delete,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Raise,
        ast.Assert,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Expr,
        ast.Pass,
        ast.Break,
        ast.Continue,
    )

# TODO add ast.TypeAlias to the list when python 3.13 is supported by gast


def walk(
    node: ast.AST, typecheck: type | None = None, stopTypecheck: type | None = None
):
    """
    Recursively yield all nodes matching the typecheck
    in the tree starting at *node* (B{excluding} *node* itself), in bfs order.

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
        # Dispatch the event to all the associated listeners
        if type(event) in self._events:
            listeners = self._events[type(event)]

            for listener in listeners:
                listener(event)

    def addEventListener(
        self, event_type: type[Event], listener: EventListener
    ) -> None:
        """
        Add an event listener for an event type
        """
        # Add listener to the event type
        if not self.hasListener(event_type, listener):
            listeners = self._events.get(event_type, [])
            listeners.append(listener)
            self._events[event_type] = listeners

    def removeEventListener(
        self, event_type: type[Event], listener: EventListener
    ) -> None:
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

    mod: "Module"


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

    mod: "Module"


@dataclasses.dataclass(frozen=True)
class ModuleRemovedEvent(Event):
    """
    When a module is removed from the passmanager.
    """

    mod: Module


@dataclasses.dataclass(frozen=True)
class RunningTransform(Event):
    """
    Before a transformation is run. 
    """
    
    transformation: type[Transformation]
    node: ast.AST


@dataclasses.dataclass(frozen=True)
class TransformEnded(Event):
    """
    After a transformation has been run.
    """
    
    transformation: type[Transformation]
    node: ast.AST


@dataclasses.dataclass(frozen=True)
class RunningAnalysis(Event):
    """
    Before an analysis is run. 
    """
    
    analysis: type[Analysis]
    node: ast.AST


@dataclasses.dataclass(frozen=True)
class AnalysisEnded(Event):
    """
    After an analysis has been run.
    """
    
    analysis: type[Analysis]
    node: ast.AST
    result: _AnalysisResult


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
    In charge of finding python modules and creating L{Module} instances for them.
    """

    search_context: SearchContext

    def module_by_name(self, modname: str) -> Module:
        """
        Find a module by name based on the current search context.
        """

    def modules_by_path(self, path: Path) -> Iterator[Module]:
        """
        If the path points to a directory it will yield recursively
        all modules under the directory. Pass a file and it will alway yield one Module entry.
        """


class _Node2RootMapping(Mapping[ast.AST, ast.Module]):
    """
    Tracks the root modules of all nodes in the system.

    Part of L{ModuleCollection}. 
    """
    def __init__(self, dispatcher: EventDispatcher) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleChangedEvent, self._onModuleChangedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[ast.AST, ast.Module] = {}

    def _onModuleAddedEvent(self, event: ModuleAddedEvent | ModuleChangedEvent) -> None:
        newmod = event.mod.node
        for node in ast.walk(newmod):
            self.__data[node] = newmod

    def _onModuleRemovedEvent(
        self, event: ModuleRemovedEvent | ModuleChangedEvent
    ) -> None:
        # O(n), every time :/
        node = event.mod.node
        to_remove = []
        for n, r in self.__data.items():
            if r is node:
                to_remove.append(n)
        for n in to_remove:
            del self.__data[n]

    def _onModuleChangedEvent(self, event: ModuleChangedEvent) -> None:
        self._onModuleRemovedEvent(event)
        self._onModuleAddedEvent(event)

    # Boring mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: ast.AST) -> ast.Module:
        return self.__data[__key]

    def __iter__(self) -> Iterator[ast.AST]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class ModuleCollection(Mapping[str | ast.Module | ast.AST, Module]):
    """
    A fake sys.modules to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher):
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[ast.Module, Module] = {}
        self.__roots = _Node2RootMapping(dispatcher)

        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if self.get(modname) not in (None, mod):
            raise ValueError(
                f"a module named {modname!r} " f"already exist: {self[modname]}"
            )

        if self.get(modnode) not in (None, mod):
            raise ValueError(
                f"the ast of the module {modname!r} is already "
                f"associated with another module: {self[modnode]}"
            )

        # register the module as beeing a part of this collection.
        self.__name2module[modname] = mod
        self.__node2module[modnode] = mod

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if modname not in self or modnode not in self:
            raise ValueError(f"looks like this module is not in the collection: {mod}")

        # remove the module from the collection
        del self.__name2module[modname]
        del self.__node2module[modnode]

    # Mapping interface

    def __getitem__(self, __key: str | ast.Module | ast.AST) -> Module:
        if isinstance(__key, str):
            return self.__name2module[__key]
        elif isinstance(__key, ast.Module):
            return self.__node2module[__key]
        elif isinstance(__key, ast.AST):
            return self.__node2module[self.__roots[__key]]
        else:
            raise TypeError(f"unexpected key type: {__key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__name2module)

    def __len__(self) -> int:
        return len(self.__name2module)


class _AnalysisResult:
    """
    Simple wrapper for the result of an analysis.
    """
    @property
    def result(self) -> object:
        raise NotImplementedError(self.result)

    @classmethod
    def Success(self, result):
        return _AnalysisSuccess(result)

    @classmethod
    def Error(self, exception):
        return _AnalysisError(exception)


@dataclasses.dataclass(frozen=True)
class _AnalysisError(_AnalysisResult):
    _error: Exception

    @property
    def result(self):
        raise self._error


@dataclasses.dataclass(frozen=True)
class _AnalysisSuccess(_AnalysisResult):
    _result: object

    @property
    def result(self):
        return self._result


class _AnalysisCache:
    """
    The strucutre of the cache consist in nested dicts.
    But this class facilitates the messages with the module pass manager.
    """

    def __init__(self) -> None:
        self.__data: dict[type[Analysis], dict[ast.AST, _AnalysisResult]] = defaultdict(
            dict
        )

    def set(self, analysis: type[Analysis], node: ast.AST, result: _AnalysisResult):
        """
        Store the analysis result in the cache.
        """
        self.__data[analysis][node] = result

    def get(self, analysis: type[Analysis], node: ast.AST) -> Optional[_AnalysisResult]:
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

    def analysisTypes(self) -> Iterator[type[Analysis]]:
        yield from self.__data


T = TypeVar("T")
RunsOnT = TypeVar("RunsOnT")
ReturnsT = TypeVar("ReturnsT")
ChildAnalysisReturnsT = TypeVar("ChildAnalysisReturnsT")


class _PassMeta(type):
    # The meta type is only necessary to implement the subclassing with arguments
    # feature of the Pass class.
    # We should not add methods to the meta type unless absolutely required.
    # This hack limits the subclass explosition issue, but not really;
    # by creating cached subclasses when calling the class with arguments.

    # TODO: required parameter should be supported as positional arguments.
    # but we need to handle when there are several values for one argument.
    @overload
    def __call__(cls: type[Pass]) -> Pass:...
    @overload
    def __call__(cls: type[Pass], **kwargs: Hashable) -> type[Pass]:...
    def __call__(cls: type[Pass], **kwargs: Hashable) -> Pass | type[Pass]:
        if not kwargs:
            # create instance only when calling with no arguments.
            return super().__call__()

        # otherwise create a derived type that binds the class attributes.
        # This is how we create analysis with several options.
        return cls._newTypeWithOptions(**kwargs)


class _PassDependencyDescriptor:
    """
    Simple container for a callback. 
    We kinda re-implement part of the descriptor protocol here.

    @see: L{Pass.__getattribute__}
    """
    def __init__(self, callback: Callable[[], Any]) -> None:
        self.callback = callback


class Pass(Generic[RunsOnT, ReturnsT], metaclass=_PassMeta):
    """
    The base class for L{Analysis} and L{Transformation}.
    """
    
    dependencies: Sequence[type[Pass]] = ()
    """
    Statically declared dependencies: If your pass requires a previous pass list it here. 
    All L{Transformation}s will be applied eagerly while L{Analysis} result will be wrapped in a descriptor
    waiting to be accessed before running. 
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
    def _newTypeWithOptions(cls, **kwargs:Hashable):
        # verify no junk arguments are slipping throught.
        validNames = ordered_set((*cls.requiredParameters, *cls.optionalParameters))
        try:
            junk = next(p for p in kwargs if p not in validNames)
        except StopIteration:
            pass
        else:
            raise TypeError(f"{cls.__qualname__}() does not recognize keyword {junk!r}")

        # remove the arguments that are already set to their values.
        kwargs = {k: v for k, v in kwargs.items() if getattr(cls, k, object()) != v}

        # if any of the arguments are not set to thei default values (or if a required argument is already set),
        # it means we're facing a case of doudle calling the class: this is not supported and we raise an error.
        # it's supported because it might creates two different classes that do exactly the same thing :/
        _nothing = object()
        if any(
            getattr(cls, promlematic := a, _nothing) is not _nothing
            for a in cls.requiredParameters
        ) or any(
            getattr(cls, promlematic := a, _d) is not _d
            for a, _d in cls.optionalParameters.items()
        ):
            hint = ""
            if cls.__qualname__ != cls.__bases__[0].__qualname__:
                hint = " (hint: analysis subclassing is not supported)"

            raise TypeError(
                f"Specifying parameter {promlematic!r} this way is not supported{hint}, "
                f"you must list all parameters into a single call to class {cls.__bases__[0].__qualname__}"
            )

        # This will automatically trigger __init_subclass__, but just once because the resulting class is cached.
        return _newsubclass(cls, **kwargs)

    def __init_subclass__(cls, **kwargs:Hashable):
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
            clash = next(
                p
                for p in (d.__name__ for d in cls.dependencies)
                if getattr(cls, p, _nothing) is not _nothing
            )
        except StopIteration:
            pass
        else:
            raise TypeError(
                f"{cls.__qualname__}: invalid class declaration, name {clash!r} is already taken by a dependency"
            )

    @classmethod
    @lru_cache(maxsize=None)
    def _getAllDependencies(cls) -> Collection[type[Pass]]:
        seen = ordered_set()
        def _yieldDeps(c:type[Analysis]):
            yield from (d for d in c.dependencies if d not in seen)
            yield from (d for d in chain.from_iterable(_yieldDeps(dep) for dep in c.dependencies) if d not in seen)
        seen.update(_yieldDeps(cls))
        return tuple(seen) 
    
    @classmethod
    def _isAbstract(cls) -> bool:
        try:
            cls._verifyRequiredParamaters()
        except TypeError:
            return True
        if cls.doPass is Pass.doPass:
            return True
        return False

    @classmethod
    @lru_cache(maxsize=None)
    def _usesModulesTransitive(cls) -> bool:
        """
        Whether this analysis uses the L{modules}.
        """
        return any(d is modules for d in cls._getAllDependencies())

    def __getattribute__(self, name):
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            return attr.callback()
        return attr

    def _verifyDependencies(self):
        pass
        # Any kinf of analsis can depend on any other at the moment.

    @classmethod
    def _verifyRequiredParamaters(cls):
        # verify required arguments exists
        _nothing = object()
        try:
            missing = next(
                p
                for p in cls.requiredParameters
                if getattr(cls, p, _nothing) is _nothing
            )
        except StopIteration:
            pass
        else:
            raise TypeError(f"{cls.__qualname__}() is missing keyword {missing!r}")

    def prepare(self, node: RunsOnT):
        """
        Gather analysis result required by this analysis.
        """
        self._verifyDependencies()
        self._verifyRequiredParamaters()
        
        # Apply all transformations eagerly, since we use a descriptor for all analyses
        # we need to transitively iterate dependent transforms and apply then now.
        for transformation in self._getAllDependencies():
            if issubclass(transformation, Transformation):
                # Transformations always run the module since there
                # are only module wide transformations at the moment.
                self.passmanager.apply(transformation, self.passmanager.module.node)

        for analysis in self.dependencies:
            if issubclass(analysis, Transformation):
                pass
            elif issubclass(analysis, Analysis):
                gather_on_node = node
                if issubclass(analysis, ModuleAnalysis):
                    gather_on_node = self.passmanager.module.node

                # Use a descriptor-like interfacefor all non-transformations.
                # If the dependency is abstract (i.e it does not have a required parameter)
                # the dependent analyses should not use the self.analysis_name attribute - 
                # it's going to fail with TypeError :/
                # And instead use self.gather(analysis_name(keyword='stuff'), node)
                callback = partial(
                    self.passmanager.gather, analysis, gather_on_node
                )  # type:ignore
                setattr(self, analysis.__name__, _PassDependencyDescriptor(callback))
            else:
                raise TypeError(
                    f"dependencies should be a Transformation or an Analysis, not {analysis}"
                )

    def _attach(self, pm: ModulePassManager):
        # Since a pass will only be instantiated with no arguments,
        # we need this extra method to set the pass manager.
        self.passmanager = pm

    def run(self, node: RunsOnT) -> ReturnsT:
        """
        Compute the analysis results. 
        If self is transformation, apply transform and return new node.

        Override this to add special pre or post processing handlers.
        """
        self.prepare(node)
        return self.doPass(node)

    def apply(self, node: RunsOnT) -> tuple[bool, ReturnsT]:
        """
        Apply transformation and return if an update happened.
        If self is an analysis, print results.
        """
        raise NotImplementedError(self.apply)

    def doPass(self, node: RunsOnT) -> ReturnsT:
        """
        Override this to add actual pass logic.
        """
        raise NotImplementedError(self.doPass)


class Analysis(Pass[RunsOnT, ReturnsT]):
    """
    A pass that does not change its content but gathers informations about it.

    An analysis can have transformations listed in it's dependencies 
    but should not call C{self.passmanager.apply} and must not manually update the tree.
    """

    update = False  # TODO: Is this needed?

    def run(self, node: RunsOnT) -> ReturnsT:
        typ = type(self)
        self.passmanager._dispatcher.dispatchEvent(
            RunningAnalysis(typ, node)
        )
        try:
            analysis_result = self.passmanager.cache.get(typ, node)
            if analysis_result is not None:  # the result is cached
                result = (
                    analysis_result.result
                )  # will rase an error if the initial analysis raised
            else:
                try:
                    # this will call prepare().
                    result = super().run(node)
                except Exception as e:
                    analysis_result = _AnalysisResult.Error(e)
                    self.passmanager.cache.set(typ, node, analysis_result)
                    raise
                analysis_result = _AnalysisResult.Success(result)
                if not isinstance(self, DoNotCacheAnalysis):
                    # only set values in the cache for non-proxy analyses.
                    self.passmanager.cache.set(
                        typ, node, analysis_result
                    )
        finally:
            self.passmanager._dispatcher.dispatchEvent(
                AnalysisEnded(typ, node, analysis_result)
            )
        return result

    def apply(self, node: RunsOnT) -> tuple[bool, RunsOnT]:
        print(self.run(node))
        return False, node


class Transformation(Pass[ast.Module | None, ast.Module]):
    """
    A pass that updates the module's content.
    
    A transformation must never update other modules, but otherwise can change anything in
    the current module including global varibles functions and classes.
    """

    preserves_analysis = ()
    """
    One of the jobs of the PassManager is to optimize how and when analyses are run. 
    In particular, it attempts to avoid recomputing data unless it needs to. 
    For this reason, passes are allowed to declare that they preserve (i.e., they don’t invalidate) an 
    existing analysis if it’s available. For example, a simple constant folding pass would not modify the 
    CFG, so it can’t possibly affect the results of dominator analysis. 
    
    By default, all passes are assumed to invalidate all others in the current module as
    well as all other analyses that transitively uses the L{modules} analisys.

    This variable should be overridden by subclasses to provide specific list.
    """

    update = False
    """
    It should be True if the module was modified by the transformation and False otherwise.
    """

    def run(self, node: ast.Module | None = None) -> ast.Module:
        typ = type(self)
        if node is None:
            node = self.passmanager.module.node
        else:
            assert node is self.passmanager.module.node
        # TODO: think of a way to cache no-op transformations so that they are not re-run every time an analysis 
        # with transformations in it's dependencies is ran...
        self.passmanager._dispatcher.dispatchEvent(RunningTransform(typ, node))
        try:
            # TODO: Does it actually fixes the new node locations ? I don't think so.
            n = super().run(node)
            # the transformation updated the AST, so analyses may need to be rerun
            if self.update:
                self.passmanager._moduleTransformed(self, self.passmanager.module)
        finally:
            self.passmanager._dispatcher.dispatchEvent(TransformEnded(typ, node))

        return n

    def apply(self, node: ast.Module | None = None) -> tuple[bool, ast.Module]:
        new_node = self.run(node)
        return self.update, new_node


class ModuleAnalysis(Analysis[ast.Module, ReturnsT]):

    """An analysis that operates on a whole module."""


class FunctionAnalysis(
    Analysis[Union[ast.FunctionDef, ast.AsyncFunctionDef], ReturnsT]
):

    """An analysis that operates on a function."""


class ClassAnalysis(Analysis[ast.ClassDef, ReturnsT]):

    """An analysis that operates on a class."""


class NodeAnalysis(Analysis[ast.AST, ReturnsT]):

    """An analysis that operates on any node."""


class DoNotCacheAnalysis(Analysis[RunsOnT, ReturnsT]):
    
    """
    An analysis that will never be cached.
    """


class _AnalysisProxy(
    DoNotCacheAnalysis[ast.Module, Mapping["_FuncOrClassTypes", ChildAnalysisReturnsT]],
    Generic[ChildAnalysisReturnsT],
):
    """
    A module analysis that returns a simple structure proxy for containing class or function nodes.
    """

    # the results of each analysis and make them accessible in the result dict proxy.
    # the result must watch invalidations in order to reflect the changes

    requiredParameters = ("analysis",)

    if TYPE_CHECKING:
        analysis: type[Analysis]

    def __keys_factory(self, node: ast.Module) -> Collection[_FuncOrClassTypes]:
        if isinstance(self.analysis, ClassAnalysis):
            typecheck = (ast.ClassDef,)
        elif isinstance(self.analysis, FunctionAnalysis):
            typecheck = (ast.FunctionDef, ast.AsyncFunctionDef)
        else:
            raise NotImplementedError()

        return ordered_set(
            walk(
                node,
                typecheck,
                # For performance, we stop the tree walking as soon as we hit
                # statement nodes that cannot contain class or functions.
                _CannotContainClassOrFunctionTypes,
            )
        )

    def doPass(
        self, node: ast.Module
    ) -> Mapping[_FuncOrClassTypes, ChildAnalysisReturnsT]:
        assert isinstance(node, ast.Module)
        return _AnalysisProxyResult(
            self.passmanager, node, self.analysis, self.__keys_factory
        )


class _AnalysisProxyResult(
    Mapping["_FuncOrClassTypes", ChildAnalysisReturnsT], Generic[ChildAnalysisReturnsT]
):
    # used for class/function to module analysis promotions

    def __init__(
        self,
        passmanager: ModulePassManager,
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

    def keys(self) -> Collection[_FuncOrClassTypes]:  # type:ignore[override]
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


class modules(DoNotCacheAnalysis[None, ModuleCollection]):
    """
    Special analysis that results in the mapping of modules: L{ModuleCollection}.
    Use this to access other modules in the system.
    """

    def run(self, _: None) -> Any:
        # skip prepare() since this analysis is trivially special
        return self.doPass(None)

    def doPass(self, _: None) -> ModuleCollection:
        return self.passmanager._getModules(self)


class ModulePassManager:
    """
    Front end to the pass system when accessing L{self.passmanager <Pass.passmanager>} 
    from an analysis or transformation methods.
    """

    def __init__(self, module: Module, passmanager: PassManager) -> None:
        "constructor is private"
        self.module = module
        self.cache = _AnalysisCache()
        self._dispatcher = passmanager._dispatcher
        self._dispatcher.addEventListener(
            InvalidatedAnalysisEvent, self._onInvalidatedAnalysisEvent
        )
        self.__pm = passmanager

    def gather(self, analysis: type[Analysis], node: ast.AST | None = None):
        """
        Call an ``analysis`` on any node in the system. If the node os not given the current module is used.
        """

        if not issubclass(analysis, Analysis):
            raise TypeError(f"unexpected analysis type: {analysis}")

        if self.__pm.modules[node] is not self.module:
            return self.__pm.gather(analysis, node)

        if node is None:
            node = self.module.node

        # Promote the analysis if necessary
        if isinstance(node, ast.Module):
            # Node analysis cannot be prpoted because they can also run on modules.
            # TODO: we could introduce an ExpressionAnalysis that could be promoted to module analysis.
            if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
                # scope to module promotions
                analysis = _AnalysisProxy(analysis=analysis)
                assert issubclass(analysis, Analysis)

        a = analysis()
        a._attach(self)
        ret = a.run(node)
        return ret

    def apply(self, transformation: type[Pass], node: ast.Module | None = None):
        """
        Call a `transformation' on a `node'. If the node os not given the current module is used.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        """

        if not issubclass(transformation, (Transformation, Analysis)):
            raise TypeError(f"unexpected analysis type: {transformation}")

        if self.__pm.modules[node] is not self.module:
            return self.__pm.apply(transformation, node)

        if node is None:
            node = self.module.node

        if not isinstance(node, ast.Module):
            raise TypeError(f"unexpected node type: {node}")

        a = transformation()
        a._attach(self)
        ret = a.apply(node)
        return ret

    def _moduleTransformed(self, transformation: Transformation, mod: Module):
        """
        Alert that the given module has been transformed, this is automatically called
        at the end of a transformation if it updated the module.
        """
        self.__pm._dispatcher.dispatchEvent(  # this is for the root modules mapping.
            ModuleChangedEvent(mod)
        )

        # the transformation updated the AST, so analyses may need to be rerun
        # Instead of clearing the entire cache, only invalidate analysis that are affected
        # by the transformation.
        invalidated_analyses: set[type[Analysis]] = ordered_set()
        for mpm, analysis in self.__pm._caches.managersAnalyses():
            if (
                # if the analysis is explicitely presedved by this transform,
                # do not invalidate.
                (analysis not in transformation.preserves_analysis)
                and (
                    # if it's not explicately preserved and the transform affects the module
                    # invalidate.             or if the analysis requires other modules
                    (mpm.module.node is mod.node)
                    or (analysis._usesModulesTransitive())
                )
            ):
                invalidated_analyses.add(analysis)

        for analys in invalidated_analyses:
            # alert that this analysis has been invalidated
            self.__pm._dispatcher.dispatchEvent(
                InvalidatedAnalysisEvent(analys, mod.node)
            )

    def _onInvalidatedAnalysisEvent(self, event: InvalidatedAnalysisEvent):
        """
        Clear the cache from this analysis.
        """
        # cases:
        # NodeAnalysis
        # FunctionAnalysis
        # ClassAnalysis
        # ModuleAnalysis

        analysis: type[Analysis] = event.analysis
        node: ast.Module = event.node

        assert isinstance(node, ast.Module)
        # It cannot be an adaptor because we don't store them in the cache
        assert not issubclass(analysis, DoNotCacheAnalysis)

        if analysis._usesModulesTransitive() or node is self.module.node:
            self.cache.clear(analysis)

    def _getModules(self, analysis: Analysis) -> ModuleCollection:
        if not isinstance(analysis, modules):
            raise RuntimeError(
                f"Only the analysis {modules.__qualname__!r} can access the ModuleCollection, please use that."
            )
        return self.__pm.modules


class _PassManagerCollection(Mapping[Module, ModulePassManager]):
    """
    Mapping from L{Module} to their L{ModulePassManager}.
    """
    def __init__(self, passmanager: PassManager) -> None:
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

    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        self.__data[event.mod] = ModulePassManager(event.mod, self._passmanager)
        
        # dispatch InvalidatedAnalysisEvent for all analyses, this will only remove analyses that
        # depends on other modules
        for a in self._passmanager._caches.allAnalyses():
            if a._usesModulesTransitive():
                self._passmanager._dispatcher.dispatchEvent(
                    InvalidatedAnalysisEvent(a, event.mod.node)
                )

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        del self.__data[event.mod] # this **should** free the memory from all analyses in the cache for this module.
        
        # dispatch InvalidatedAnalysisEvent for all analyses, this will only remove analyses that
        # depends on other modules
        for a in self._passmanager._caches.allAnalyses():
            if a._usesModulesTransitive():
                self._passmanager._dispatcher.dispatchEvent(
                    InvalidatedAnalysisEvent(a, event.mod.node)
                )

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: Module) -> ModulePassManager:
        return self.__data[__key]

    def __iter__(self) -> Iterator[Module]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class _PMCacheProxy:
    """
    Provide methods to acces cached analyses types accros all L{ModulePassManager}s.
    """
    def __init__(self, pms: _PassManagerCollection) -> None:
        self.__pms = pms
    
    def managersAnalyses(self) -> Iterator[tuple[ModulePassManager, type[Analysis]]]:
        """
        For each analysis in each passmanagers yield C{(ModulePassManager, type[Analysis])}.
        """
        for mpm in self.__pms.values():
            for analysis in mpm.cache.analysisTypes():
                yield mpm, analysis
    
    def allAnalyses(self) -> Collection[type[Analysis]]:
        """
        Collect all analyses types in a set.
        """
        r = ordered_set()
        for _, a in self.managersAnalyses():
            r.add(a)
        return r

class PassManager:
    """
    Front end to the inter-modules pass system.
    One `PassManager` can be used for the analysis of a collection of modules.
    """

    def __init__(self):
        d = EventDispatcher()

        self.modules = ModuleCollection(d)
        self._dispatcher = d
        self._passmanagers = pms = _PassManagerCollection(self)
        self._caches = _PMCacheProxy(pms)

    def add_module(self, mod: Module):
        """
        Adds a new module to the pass manager.
        Use PassManager.modules to access modules.
        """
        self._dispatcher.dispatchEvent(ModuleAddedEvent(mod))

    def remove_module(self, mod: Module):
        """
        Remove a module from the passmanager.
        This will allow adding another module with the same name or same module node.
        """
        self._dispatcher.dispatchEvent(ModuleRemovedEvent(mod))

    # How to handle cycles ? With a context manager that will push onto a set of running analyses.
    @overload
    def gather(
        self, analysis: type[Analysis[RunsOnT, ReturnsT]], node: RunsOnT
    ) -> ReturnsT:
        ...

    @overload
    def gather(
        self,
        analysis: type[ClassAnalysis[ReturnsT] | FunctionAnalysis[ReturnsT]],
        node: ast.Module,
    ) -> Mapping[_FuncOrClassTypes, ReturnsT]:
        ...

    def gather(
        self, analysis: type[Analysis[RunsOnT, ReturnsT]], node: RunsOnT | ast.Module
    ) -> ReturnsT | Mapping[_FuncOrClassTypes, ReturnsT]:
        """
        High-level function to call an ``analysis`` on any node in the system.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.gather(analysis, node)

    def apply(self, transformation, node):
        """
        High-level function to call a `transformation' on a `node'.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.apply(transformation, node)


class Statistics:
    def __init__(self, dispatcher: EventDispatcher) -> None:
        pass

    def _onRun(self, event):...
    def _onFinish(self, event):...