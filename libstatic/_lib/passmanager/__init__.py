"""
This module provides a framework for L{ast} pass management. 
It's a building block to write static analyzer or compiler for Python. 

There are two kinds of passes: transformations and analysis.
    * L{ModuleAnalysis} and L{NodeAnalysis} are to be
      subclassed by any pass that collects information about the AST.
    * L{PassManager.gather} is used to gather (!) the result of an analyses on an AST node.
    * L{Transformation} is to be sub-classed by any pass that updates the AST.
    * L{PassManager.apply} is used to apply (sic) a transformation on an AST node.

To write an analysis: 

    - Subclass one of the analysis class cited above.
    - List passes required by yours in the L{Analysis.dependencies} tuple, 
      they will be built automatically and stored in the attribute with the corresponding name.
    - Write your analysis logic inside the L{Analysis.doPass} method. The analysis result must be returned by
      the L{Analysis.doPass} method or an exeception raised.
    - Use it either from another pass’s C{dependencies}, or through the L{PassManager.gather} function.

B{Example}: In the following code snippets we'll look at how to write a type inference function that
understand literal values as well as builtins calls like L{list()} with a relatively high level of confidence
we're not misinterpreting a symbol name.   

Start by coding the logic of your lower-level analysis, which in our case computes the locals for a given scope:

>>> import ast
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

>>> from .. import analyses

Note how the analyses can both be used as instance attribute, in which case they are run on the current node; 
or as C{self.passmanager.gather()} argument to run them on any other nodes.

>>> ast_scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, 
...     ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Lambda, ast.Module)
>>> class declared_names(NodeAnalysis[set[str]]):
...     "Collects all the declared names accessible from the current node"
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

Looks good... Now putting everything together in the C{infer_type} analysis:

>>> supported_builtins_calls = { # we limit ourselves to a subset of builtins calls.
...     'list': list, 
...     'dict': dict, 
...     'set': set, 
...     'tuple': tuple, 
...     'str': str, 
...     'frozenset': frozenset, 
...     'int': int
... } # that's not exhaustive but it's a start

>>> class infer_type(NodeAnalysis[type]):
...     '''
...     This class tries to infer the types of literal expression as 
...     well builtins calls like dict(). 
...     '''
...     dependencies = (declared_names, )
...     def doPass(self, node) -> type | None:
...         if (isinstance(node, ast.Call) and 
...             isinstance(node.func, ast.Name) and 
...             node.func.id in supported_builtins_calls):
...             if node.func.id not in self.declared_names:
...                 return supported_builtins_calls[node.func.id]
...             return None # the builtin name is actually shadowed so we don't know...
...         try:
...             # try the node as a literal expression
...             return type(ast.literal_eval(node))
...         except Exception:
...             return None

>>> testintcall = next(n for n in ast.walk(testnode) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'int')
>>> testsetcall = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'set')
>>> testlistcall1 = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testlistcall2 = next(n for n in ast.walk(testmethod) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testliteralstring = next(n for n in ast.walk(testnode) if isinstance(n, ast.Constant) and isinstance(n.value, str))

We now have a manner to statically infer types of builtins calls with relatively high confidence 
(It could be improved, this is just an example after all).

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

Sometime it is practical for an analysis to have different behaviour depending on optional or required parameters.
Instead of writing several L{Analysis} subclass calling un underlying function with different parameters, write an analysis taking parameters: 
    
    - List all required parameters names in the C{requiredParameters} tuple.
    - List all optional parameters in the C{optionalParameters} dict; 
      keys are parameters names are values are the default values in case none is provided.
    - To create your parameterized analysis simply call your analysis class with keywords coresponding to your
      parameters. This will create a devired subclass with the parameters set at class level.
    
To write a transformation...

    - Subclass L{Transformation} class. Keep in mind that a transformation is always run at the module level.
    - List passes required by yours in the L{Analysis.dependencies} tuple, 
      they will be built automatically and stored in the attribute with the corresponding name.
    - Write your analysis logic inside the L{Analysis.doPass} method. The analysis result must be returned by
      the L{Analysis.doPass} method or an exeception raised.
    - Use it either from another pass’s C{dependencies}, or through the L{PassManager.gather} function.


    
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
from contextlib import contextmanager
from functools import lru_cache, partial
from itertools import chain

import sys
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
    Protocol,
    TypeVar,
    Generic,
    Sequence,
    Union,
    overload,
)

import dataclasses

from beniget.ordered_set import ordered_set

from .events import (EventDispatcher, ClearAnalysisEvent, ModuleAddedEvent, 
                     ModuleChangedEvent, ModuleRemovedEvent, AnalysisEnded, 
                     RunningAnalysis, SupportLibraryEvent, TransformEnded, RunningTransform)

__docformat__ = 'epytext'

ModuleNode = Hashable
"""
Symbol that represent a ast module node. 
"""

AnyNode = Hashable
"""
Symbol that represent any kind of ast node. 
"""

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
    # assert newcls.__name__ == cls.__name__
    newcls.__qualname__ = cls.__qualname__
    assert isinstance(newcls, type)
    return newcls


@dataclasses.dataclass(frozen=True)
class Module:
    """
    The specifications of a python module.
    """

    node: ModuleNode
    """
    The module node.
    """
    
    modname: str
    """
    The module fully qualified name. 
    If the module is a package, do not include C{__init__}
    """
    
    filename: str | None = None
    """
    The filename of the source file.
    """
    
    is_package: bool = False
    """
    Whether the module is a package.
    """
    
    # TODO: namespace packages are not supported at the moment.
    # is_namespace_package: bool = False
    # """
    # Whether the module is a namespace package.
    # """
    
    is_stub: bool = False
    """
    Whether the module is a stub module.
    """
    
    code: str | None = None
    """
    The source.
    """


class _Node2RootMapping(Mapping[AnyNode, ModuleNode]):
    """
    Tracks the root modules of all nodes in the system.

    Part of L{ModuleCollection}. 
    """
    def __init__(self, dispatcher: EventDispatcher, astsupport: ASTCompat) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleChangedEvent, self._onModuleChangedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[AnyNode, ModuleNode] = {}
        self._ast = astsupport

    def _onModuleAddedEvent(self, event: ModuleAddedEvent | ModuleChangedEvent) -> None:
        newmod = event.mod.node
        # O(n), every time :/
        for node in self._ast.walk(newmod):
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
        # TODO (optimizations): 2xO(n), every time: Thid could be improved by introducing 'uptdates_regions' Transformation
        # attribute that will contain a sequence of all nodes added in the tree, we also would need a sequnce
        # aof nodes removed from the tree. 
        self._onModuleRemovedEvent(event)
        self._onModuleAddedEvent(event)

    # Boring mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: AnyNode) -> ModuleNode:
        return self.__data[__key]

    def __iter__(self) -> Iterator[AnyNode]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class ModuleCollection(Mapping[str | ModuleNode | AnyNode, Module]):
    """
    A fake C{sys.modules} to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher, astsupport: ASTCompat):
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[ModuleNode, Module] = {}
        self.__roots = _Node2RootMapping(dispatcher, astsupport)

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

    def __getitem__(self, __key: str | ModuleNode | AnyNode) -> Module:
        if isinstance(__key, str):
            return self.__name2module[__key]
        try:
            return self.__node2module[__key]
        except KeyError:
            try: 
                return self.__node2module[self.__roots[__key]]
            except KeyError:
                pass
        raise KeyError(__key)

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
    
    def isError(self) -> bool:
        return isinstance(self, _AnalysisError)


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
        self.__data: dict[type[Analysis], dict[Hashable, _AnalysisResult]] = defaultdict(
            dict
        )

    def set(self, analysis: type[Analysis], node: Hashable, result: _AnalysisResult):
        """
        Store the analysis result in the cache.
        """
        assert not analysis.do_not_cache
        self.__data[analysis][node] = result

    def get(self, analysis: type[Analysis], node: Hashable) -> Optional[_AnalysisResult]:
        """
        Query for the cached result of this analysis.
        """
        if analysis in self.__data:
            try:
                return self.__data[analysis][node]
            except KeyError:
                return None
        return None

    def clear(self, analysis: type[Analysis]):
        """
        Get rid of the the given analysis result.
        """
        if analysis in self.__data:
            del self.__data[analysis]

    def analysisTypes(self) -> Iterator[type[Analysis]]:
        """
        Get an iterator on all analyses types in this cache.
        """
        yield from self.__data


T = TypeVar("T")
RunsOnT = TypeVar("RunsOnT", bound=Hashable)

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

    # Custom repr so we can showcase the parameter values.
    def __str__(self: type[Pass]) -> str:
        passname = self.__qualname__
        args = []
        for p in self.requiredParameters:
            args.append( f'{p}={getattr(self, p, '<?>')}' )
        for p in self.optionalParameters:
            args.append( f'{p}={getattr(self, p, '<?>')}' )
        if args:
            passname += f'({", ".join(args)})'
        return f'<class {passname!r}>'
    
    __repr__ = __str__
            

class _PassDependencyDescriptor:
    """
    Simple container for a callback. 
    We kinda re-implement part of the descriptor protocol here.

    @see: L{Pass.__getattribute__}
    """
    def __init__(self, callback: Callable[[], Any]) -> None:
        self.callback = callback


class PassContext(object):

    """
    Class that does the book-keeping of the chains of passes runs.
    """
    
    def __init__(self) -> None:
        self._stack: set[tuple[type[Pass], AnyNode]] = ordered_set()

    @property
    def current(self) -> AnyNode | None:
        try:
            _, node = next(reversed(self._stack))
            return node
        except StopIteration:
            return None

    @contextmanager
    def pushPass(self, passs: type[Pass], node: AnyNode) -> Iterator[None]:
        key = (passs, node)
        if key in self._stack:
            raise RuntimeError(f'cycle detected with pass: {key}')
        self._stack.add(key)
        yield
        # pop last element
        e = next(reversed(self._stack))
        if e is not key:
            raise RuntimeError(f'pass context is confused: {e} is not {key}')
        self._stack.discard(e)


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

    requiredParameters: tuple[Hashable] | tuple[()] = ()
    """
    Some passes looks for specific name that needs to be provided as arguments. 
    """

    optionalParameters: dict[str, Hashable] = {}
    """
    Other optional arguments to their default values.
    """

    update = False
    """
    Since there is nothing that technically avoids calling L{apply} on an analysis, this flag must be set to L{False} here
    and potentially overriden in transformations.
    """

    passmanager: ModulePassManager
    ctx: PassContext

    @classmethod
    def _verifyNoJunkParameters(cls, **kwargs:Hashable):
        # verify no junk arguments are slipping throught.
        validNames = ordered_set((*cls.requiredParameters, *cls.optionalParameters))
        try:
            junk = next(p for p in kwargs if p not in validNames)
        except StopIteration:
            pass
        else:
            raise TypeError(f"{cls.__qualname__}() does not recognize keyword {junk!r}")

    @classmethod
    def _verifyParametersOlderValues(cls):
        # if any of the arguments are not set to their default values (or if a required argument is already set),
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

    @classmethod
    def _newTypeWithOptions(cls, **kwargs:Hashable):
        cls._verifyNoJunkParameters(**kwargs)
        cls._verifyParametersOlderValues()
        
        # Remove the arguments that are already set to their values.
        kwargs = {k: v for k, v in kwargs.items() if getattr(cls, k, object()) != v}

        # This will automatically trigger __init_subclass__, but just once because the resulting class is cached.
        return _newsubclass(cls, **kwargs)

    def __init_subclass__(cls, **kwargs:Hashable):
        # https://docs.python.org/3/reference/datamodel.html#customizing-class-creation
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
        def _yieldDeps(c: type[Pass]):
            yield from (d for d in c.dependencies if d not in seen)
            yield from (d for d in chain.from_iterable(_yieldDeps(dep) for dep in c.dependencies) if d not in seen)
        seen.update(_yieldDeps(cls))
        return tuple(seen) 
    
    @classmethod
    @lru_cache(maxsize=None)
    def _isInterModuleAnalysis(cls) -> bool:
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

    def _attach(self, pm: ModulePassManager, ctx: PassContext):
        # Since a pass will only be instantiated with no arguments,
        # we need this extra method to set the pass instance variables
        self.ctx = ctx
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

    do_not_cache = False
    """
    This indicates the pass manager that this analysis should never be cached. 
    """

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
                    if not typ.do_not_cache:
                        self.passmanager.cache.set(typ, node, analysis_result)
                    raise
                analysis_result = _AnalysisResult.Success(result)
                if not typ.do_not_cache:
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
    
    @classmethod
    def proxy(cls) -> analysis_proxy[RunsOnT, ReturnsT]:
        """
        Derive this analysis to return a simple proxy that provide a C{get} method which trigers
        the original analysis on the given node. 
        
        This can be used to avoid calling repetitively C{self.passmanager.gather(analysis, ...)}.
        """
        return analysis_proxy(analysis=cls)


class Transformation(Pass[ModuleNode, ModuleNode]):
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

    def __init__(self,):
        self.update = False
        """
        It should be True if the module was modified by the transformation and False otherwise.
        """

        # TODO: Work in this optimization
        # self.added_nodes: Sequence[AnyNode] = ()
        # self.removed_nodes: Sequence[AnyNode] = ()

    def run(self, node: ModuleNode) -> ModuleNode:
        typ = type(self)
        if node is not self.passmanager.module.node:
            # TODO: This limitation might be lifted in the future.
            raise RuntimeError(f'invalid node: {node}, expected {self.passmanager.module.node}')
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

    def apply(self, node: ModuleNode) -> tuple[bool, ModuleNode]:
        new_node = self.run(node)
        return self.update, new_node


class ModuleAnalysis(Analysis[ModuleNode, ReturnsT]):

    """An analysis that operates on a whole module."""

class NodeAnalysis(Analysis[AnyNode, ReturnsT]):

    """An analysis that operates on any node."""

class FunctionAnalysis(Analysis[AnyNode, ReturnsT]):

    """An analysis that operates on a function."""


class ClassAnalysis(Analysis[AnyNode, ReturnsT]):

    """An analysis that operates on a class."""


class GetProxy(Generic[RunsOnT, ReturnsT]):
    """
    Provide L{get} method that defers to the given callable in the constructor.
    """
    def __init__(self, factory: Callable[[RunsOnT], ReturnsT]):
        self._factory = factory

    # TODO: write overloads for this method...
    def get(self, 
            key: RunsOnT, 
            default: Any = None, 
            suppress: bool | type[Exception] | tuple[type[Exception], ...] = False) -> ReturnsT:
        """
        Request a value from this proxy. 

        @param suppress: If suppres is a type or tuple of types, will return C{default} if an exception matching
            given types is raise during processing. If suppress is True, it will return C{default} for all L{Exception}s. 
            The default behaviour is to always raise.
        """
        try:
            return self._factory(key)
        except Exception as e:
            if suppress is not False:
                if suppress is True:
                    return default
                if isinstance(e, suppress):
                    return default
            raise


class analysis_proxy(Analysis[object, GetProxy[RunsOnT, ChildAnalysisReturnsT]], Generic[RunsOnT, ChildAnalysisReturnsT]):
    """
    An analysis that returns a simple proxy that provide a C{get} method which trigers
    the underlying analysis on the given node.
    """
    
    do_not_cache = True

    requiredParameters = ("analysis",)

    if TYPE_CHECKING:
        analysis: type[Analysis]
    
    def __init_subclass__(cls, **kwargs: Hashable):
        super().__init_subclass__(**kwargs)
        # Override the __name__ so the dependency still has the actual analysis name.
        cls.__name__ = cls.analysis.__name__

    def doPass(self, _: object) -> GetProxy:
        if issubclass(self.analysis, __class__):
            raise TypeError(f"Can't proxy an alaysis that is already a proxy: {self.analysis}")
        
        return GetProxy(partial(self.passmanager.gather, self.analysis))


# This would be called a immutable pass in the LLVM jargo.
class modules(Analysis[object, ModuleCollection]):
    """
    Special analysis that results in the mapping of modules: L{ModuleCollection}.
    Use this to access other modules in the system.
    """

    do_not_cache = True

    def run(self, _: object) -> Any:
        # skip prepare() since this analysis is trivially special
        return self.doPass(None)

    def doPass(self, _: object) -> ModuleCollection:
        return self.passmanager._getModules(self)

# TODO: We could unify ModulePassManager and PassManager classes since they are very coupled.
# and use the _VerifyCallsPassManager to ensure no modules are added or removed from within a pass.
class ModulePassManager:
    """
    Front end to the pass system when accessing L{self.passmanager <Pass.passmanager>} 
    from an analysis or transformation methods.
    """

    def __init__(self, module: Module, passmanager: PassManager) -> None:
        "constructor is private"
        self.module = module
        self.cache = _AnalysisCache()
        self._dispatcher: EventDispatcher = passmanager._dispatcher
        self._dispatcher.addEventListener(
            ClearAnalysisEvent, self._onClearAnalysisEvent
        )
        self.__pm = passmanager

    def gather(self, analysis: type[Analysis], node: AnyNode):
        """
        Call an L{analysis} on any node in the system.
        """

        if not issubclass(analysis, Analysis):
            raise TypeError(f"unexpected analysis type: {analysis}")

        if self.__pm.modules[node] is not self.module:
            return self.__pm.gather(analysis, node)

        # TODO: This feature is nice but it can be replaced by using 'analysis.proxy()' instead of 'analysis'.
        # So I don't think it's worth it given the fact we aim to be library agnostic.
        # # Promote the analysis if necessary
        # if isinstance(node, ast.Module):
        #     # Node analysis cannot be promoted because they can also run on modules.
        #     # TODO: we could introduce an ExpressionAnalysis that could be promoted to module analysis.
        #     if issubclass(analysis, (FunctionAnalysis, ClassAnalysis)):
        #         # scope to module promotions
        #         analysis = analysis_proxy(analysis=analysis)
        #         assert issubclass(analysis, Analysis)

        with self.__pm._ctx.pushPass(analysis, node):
            a = analysis()
            a._attach(_VerifyCallsPassManager(a, self), self.__pm._ctx)
            ret = a.run(node)
        
        return ret

    def apply(self, transformation: type[Pass], node: Any):
        """
        Call a C{transformation} on a module C{node}.
        """

        if not issubclass(transformation, Pass):
            raise TypeError(f"unexpected pass type: {transformation}")

        if self.__pm.modules[node] is not self.module:
            return self.__pm.apply(transformation, node)

        # if not isinstance(node, ast.Module):
        #     raise TypeError(f"unexpected node type: {node}")
        
        with self.__pm._ctx.pushPass(transformation, node):
            a = transformation()
            a._attach(_VerifyCallsPassManager(a, self), self.__pm._ctx)
            ret = a.apply(node)
        
        return ret

    def _moduleTransformed(self, transformation: Transformation, mod: Module):
        """
        Alert that the given module has been transformed, this is automatically called
        at the end of a transformation if it updated the module.
        """
        self._dispatcher.dispatchEvent(  # this is for the root modules mapping.
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
                    or (analysis._isInterModuleAnalysis())
                )
            ):
                invalidated_analyses.add(analysis)

        for analys in invalidated_analyses:
            # alert that this analysis has been invalidated
            self._dispatcher.dispatchEvent(
                ClearAnalysisEvent(analys, mod.node)
            )

    def _onClearAnalysisEvent(self, event: ClearAnalysisEvent):
        """
        Clear the cache from this analysis.
        """
        # cases:
        # NodeAnalysis
        # FunctionAnalysis
        # ClassAnalysis
        # ModuleAnalysis

        analysis: type[Analysis] = event.analysis
        node: ModuleNode = event.node

        if analysis._isInterModuleAnalysis() or node is self.module.node:
            self.cache.clear(analysis)

    def _getModules(self, analysis: Analysis) -> ModuleCollection:
        if not isinstance(analysis, modules):
            raise RuntimeError(
                f"Only the analysis {modules.__qualname__!r} can access the ModuleCollection, use that in your pass dependecies."
            )
        return self.__pm.modules

class _VerifyCallsPassManager:
    """
    A proxy to the pass manager that makes sure an intra-module analyses does never depend on inter-module analysis.
    """
    def __init__(self, analysis: Analysis, mpm: ModulePassManager) -> None:
        self.__mpm = mpm
        
        if not analysis._isInterModuleAnalysis():

            def gather(analysis_: type[Analysis], node: AnyNode):
                if analysis_._isInterModuleAnalysis():
                    raise TypeError(f'You must list {modules.__qualname__} in you pass dependencies to gather this pass: {analysis_}')    
                return mpm.gather(analysis_, node)        
            
            def apply(transformation_: type[Transformation], node: AnyNode):
                if transformation_._isInterModuleAnalysis():
                    raise TypeError(f'You must list {modules.__qualname__} in you pass dependencies to apply this pass: {transformation_}')  
                return mpm.apply(transformation_, node)        

            self.apply = apply
            self.gather = gather
    
    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            try:
                return getattr(self.__mpm, name)
            except AttributeError:
                raise e

if TYPE_CHECKING:
    _VerifyCallsPassManager = ModulePassManager

class _PassManagerCollection(Mapping[Module, ModulePassManager]):
    """
    Mapping from L{Module} to their L{ModulePassManager}.
    """
    def __init__(self, passmanager: PassManager) -> None:
        super().__init__()
        self.__pm = passmanager

        # register the event listeners
        passmanager._dispatcher.addEventListener(
            ModuleAddedEvent, self._onModuleAddedEvent
        )
        passmanager._dispatcher.addEventListener(
            ModuleRemovedEvent, self._onModuleRemovedEvent
        )

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[Module, ModulePassManager] = {}
    
    def _invalidateAllInterModulesAnalyses(self, module: Module):
        for a in self.__pm._caches.allAnalyses():
            if a._isInterModuleAnalysis():
                self.__pm._dispatcher.dispatchEvent(
                    ClearAnalysisEvent(a, module.node)
                )

    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        self.__data[event.mod] = ModulePassManager(event.mod, self.__pm)
        self._invalidateAllInterModulesAnalyses(event.mod)

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        del self.__data[event.mod] # this **should** free the memory from all analyses in the cache for this module.
        self._invalidateAllInterModulesAnalyses(event.mod)

    # Mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: Module) -> ModulePassManager:
        return self.__data[__key]

    def __iter__(self) -> Iterator[Module]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class _Caches:
    """
    Provide methods to acces cached analyses types accros all L{ModulePassManager}s.
    """
    def __init__(self, passmanagers: _PassManagerCollection) -> None:
        self.__pms = passmanagers
    
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


class ILibrarySupport(Protocol):
    """
    Instances of this class carry all the required information for the passmanager to support
    concrete types of nodes like the one created by standard library L{ast} or L{astroid} or L{gast} or L{parso}.

    Currently, the only thing that needs to be known about the AST is how to iterate across
    all the children nodes. But that list might grow with the future developments
    """

    iter_child_nodes: Callable[[AnyNode], Iterable[AnyNode]]
    """
    Callable that yields the direct child node starting at the given node inclusively. Like L{ast.iter_child_nodes}.
    If the given node is not one of the supported types, the function must raise L{NotImplementedError}.
    """


class ASTCompat:
    """
    Wrapper to support multiple concrete types of nodes based on registered strategies.
    """
    
    def __init__(self, dispatcher: EventDispatcher):
        self._supports: ordered_set[ILibrarySupport] = ordered_set()
        dispatcher.addEventListener(SupportLibraryEvent, self._onSupportLibraryEvent)


    def _onSupportLibraryEvent(self, event: SupportLibraryEvent):
        self._supports.add(event.lib)


    def iter_child_nodes(self, node: AnyNode) -> Iterable[AnyNode]:
        """
        Like L{ast.iter_child_nodes}.
        """
        for lib in self._supports:
            try:
                it = lib.iter_child_nodes(node)
            except NotImplementedError:
                continue
            else:
                return it
        raise TypeError(f'node type not supported: {node}')
    

    def walk(self, node: AnyNode, typecheck: type | None = None, stopTypecheck: type | None = None):
        """
        Recursively yield all nodes matching the typecheck
        in the tree starting at *node* (B{excluding} *node* itself), in bfs order.

        Do not recurse on children of types matching the stopTypecheck type.
        """
        from collections import deque

        todo = deque(self.iter_child_nodes(node))
        while todo:
            node = todo.popleft()
            if stopTypecheck is None or not isinstance(node, stopTypecheck):
                todo.extend(self.iter_child_nodes(node))
            if typecheck is None or isinstance(node, typecheck):
                yield node


class PassManager:
    """
    Front end to the pass system.
    One L{PassManager} can be used for the analysis of a collection of modules.
    """

    def __init__(self):
        
        self._dispatcher = d = EventDispatcher()
        self._ast = astsupport = ASTCompat(d)
        _init_support(self)

        self.modules = ModuleCollection(d, astsupport)
        """
        Contains all the modules in the system.
        """

        
        self._passmanagers = pms = _PassManagerCollection(self)
        self._caches = _Caches(pms)
        self._ctx = PassContext()

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
    # @overload
    # def gather(
    #     self, analysis: type[Analysis[RunsOnT, ReturnsT]], node: RunsOnT
    # ) -> ReturnsT:
    #     ...

    # @overload
    # def gather(
    #     self,
    #     analysis: type[ClassAnalysis[ReturnsT] | FunctionAnalysis[ReturnsT]],
    #     node: ast.Module,
    # ) -> Mapping[_FuncOrClassTypes, ReturnsT]:
    #     ...

    # def gather(
    #     self, analysis: type[Analysis[RunsOnT, ReturnsT]], node: RunsOnT | ast.Module
    # ) -> ReturnsT | Mapping[_FuncOrClassTypes, ReturnsT]:

    def gather(self, analysis: type[Analysis], node: AnyNode):
        """
        High-level function to call an L{Analysis} on any node in the system.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.gather(analysis, node)

    def apply(self, transformation: type[Transformation], node: ModuleNode):
        """
        High-level function to call a L{Transformation} on a C{node}.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.apply(transformation, node)

    def support(self, lib: ILibrarySupport) -> None:
        """
        Add support for AST parser. This should be called first if you use the pass manager with 
        a AST parser library that is not supported by default.
        """
        self._dispatcher.dispatchEvent(SupportLibraryEvent(lib))
    
def _init_support(pm: PassManager):

    # At the moment I'm writing this lines, a simple self.support(ast) would suffice
    # but in order to settle the structure for future developments each library has it's onw class.

    # Support standard library
    import ast
    class standard_lib:
        @staticmethod
        def iter_child_nodes(node:ast.AST) -> Iterable[ast.AST]:
            if not isinstance(node, ast.AST):
                raise NotImplementedError()
            return ast.iter_child_nodes(node)
    
    pm.support(standard_lib)
    
    # Support gast library
    if 'gast' in sys.modules:
        import gast
        # At the moment this strategy provides the same information as the standard library one,
        # but that might evolve in the future.
        class gast_lib:
            iter_child_nodes = standard_lib.iter_child_nodes
        
        pm.support(gast_lib)
    
    # Support parso library
    if 'parso' in sys.modules:
        import parso.tree
        class parso_lib:
            @staticmethod
            def iter_child_nodes(node):
                if isinstance(node, parso.tree.BaseNode):
                    return node.children
                raise NotImplementedError()
        
        pm.support(parso_lib)

    # Support astroid library
    if 'astroid' in sys.modules:
        import astroid
        class astroid_lib:
            @staticmethod
            def iter_child_nodes(node):
                if isinstance(node, astroid.NodeNG):
                    return node.children
                raise NotImplementedError()
        
        pm.support(astroid_lib)


class Statistics:
    def __init__(self, dispatcher: EventDispatcher) -> None:
        pass

    def _onRun(self, event):...
    def _onFinish(self, event):...