
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

# TODO: Implement complete inter-modulea analysis that would stay in the cache when a module is added. 

# TODO: Implment transform added / remove nodes in order to optimize visiting time for root module mapping

# TODO: Integrate ModulePassManager and PassManager into one API: The single PassManager has to be composed by all the cahes.
# refactor _PassManagerCollection into _AnalysisCacheCollection and _VerifyCallsPassManager into _PassManagerProxy that will also
# verify that an analysis is not adding a new module for instance, and well restrict direct access to passmanager.modules (onstead of using the _getModules hack)

# TODO: Create an adaptor class that turns a callable '(passmanager, node) -> result' into an analysis.
# TODO: Create an adaptor class that turns a callable '(passmanager, node) -> (newnode, update, preserves_analysis, added_nodes, removed_nodes)' into a transformation

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
    Self,
    TypeAlias,
    TypeVar,
    Generic,
    Sequence,
    Protocol,
    cast,
    overload,
)

from beniget.ordered_set import ordered_set # type: ignore

from .events import (EventDispatcher, ClearAnalysisEvent, ModuleAddedEvent, 
                     ModuleChangedEvent, ModuleRemovedEvent, AnalysisEnded, 
                     RunningAnalysis, SupportLibraryEvent, TransformEnded, RunningTransform)
from ._modules import Module, ModuleCollection
from ._astcompat import ASTCompat, ILibrarySupport
from ._caching import AnalysisCache, AnalysisResult

__docformat__ = 'epytext'

ModuleNode = Any
"""
Symbol that represent a ast module node. 
"""

AnyNode = Any
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




T = TypeVar("T")
RunsOnT = TypeVar("RunsOnT") 
# We should use bound=Hashable but mypy complains too much to that

if TYPE_CHECKING:
    class PassLike(Protocol, type): # type: ignore[misc]
        """
        In practice, only L{Analysis} and L{Transformation} subclasses are valid implementations of this interface.
        """
        def __call__(self) -> Pass:...
        def _isInterModuleAnalysis(self) -> bool:...

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
    def __call__(cls: type[Pass]) -> Pass:... # type: ignore
    @overload
    def __call__(cls: type[Pass], **kwargs: Hashable) -> type[Pass]:... # type: ignore
    def __call__(cls: type[Pass], **kwargs: Hashable) -> Pass | type[Pass]: # type: ignore
        if not kwargs:
            # create instance only when calling with no arguments.
            return super().__call__() # type: ignore

        # otherwise create a derived type that binds the class attributes.
        # This is how we create analysis with several options.
        return cls._newTypeWithOptions(**kwargs)

    # Custom repr so we can showcase the parameter values.
    def __str__(self: type[Pass]) -> str: # type: ignore
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
    Class that does the book-keeping of the chains of passes runs and provide
    accessors for the current node and the current modules (both might represent the same).

    It is accessible by in the L{Pass.ctx} instance attribute.
    """
    
    def __init__(self, modules: ModuleCollection) -> None:
        self._stack: ordered_set[tuple[type[Pass], AnyNode]] = ordered_set()
        self._modules = modules

    @property
    def current(self) -> AnyNode:
        try:
            _, node = next(reversed(self._stack))
            return node
        except StopIteration:
            raise RuntimeError('pass context has no state')
    
    @property
    def module(self) -> Module:
        return self._modules[self.current]

    @contextmanager
    def pushPass(self, passs: PassLike, node: AnyNode) -> Iterator[None]:
        key = (passs, node)
        if key in self._stack:
            # TODO: Use a exception subclass in order to potentially catch and
            # ase another strategy for analysing this node.
            raise RuntimeError(f'cycle detected with pass: {key}')
        self._stack.add(key)
        yield
        # pop last element
        e = next(reversed(self._stack))
        if e is not key:
            raise RuntimeError(f'pass context is confused: {e} is not {key}')
        self._stack.discard(e)


# this object does not exist at runtime it only helps
# for th typing.
class IPassManager(Protocol): 
    """
    This interface defines how to run passes.

    @note: This is the passmanager you have access to from inside a L{Pass}, 
        it is stored in the L{Pass.passmanager} instance attribute.
    """
    def apply(self, transformation: PassLike, node: AnyNode) -> tuple[bool, AnyNode]:...
    def gather(self, analysis: PassLike, node: AnyNode) -> Any:...


class IMutablePassManager(IPassManager, Protocol):
    """
    This interface defines how to mutate and access modules in the pass manager system.

    @note: This is the passmanager you get by calling L{PassManager()}.
    """

    def add_module(self, mod: Module) -> None: ...
    def remove_module(self, mod: Module) -> None: ...
    @property
    def modules(self) -> ModuleCollection:...


class Pass(Generic[RunsOnT, ReturnsT], metaclass=_PassMeta):
    """
    The base class for L{Analysis} and L{Transformation}.
    """
    
    dependencies: Sequence[type[Analysis] | type[Transformation]] = ()
    """
    Statically declared dependencies: If your pass requires a previous pass list it here. 
    All L{Transformation}s will be applied eagerly while L{Analysis} result will be wrapped in a descriptor
    waiting to be accessed before running. 
    """

    requiredParameters: tuple[str] | tuple[()] = ()
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

    if TYPE_CHECKING:
        # This is only for typing purposes, it could not work this way...
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.passmanager: IPassManager
            """
            Ref to the pass manager. 
            """
            
            self.ctx: PassContext
            """
            Ref to the pass context. 
            """

        def __call__(self) -> Self: ...

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
        Bind analysis result required by this analysis and apply transformations.
        """
        self._verifyDependencies()
        self._verifyRequiredParamaters()
        
        # Apply all transformations eagerly, since we use a descriptor for all analyses
        # we need to transitively iterate dependent transforms and apply then now.
        for transformation in self._getAllDependencies():
            if issubclass(transformation, Transformation):
                # Transformations always run the module since there
                # are only module wide transformations at the moment.
                self.passmanager.apply(transformation, self.ctx.module.node)

        for analysis in self.dependencies:
            if issubclass(analysis, Transformation):
                pass
            elif issubclass(analysis, Analysis):
                gather_on_node: Hashable = node
                if issubclass(analysis, ModuleAnalysis):
                    gather_on_node = self.ctx.module.node

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

    def _attach(self, pm: IPassManager, ctx: PassContext):
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
                # will rase an error if the initial analysis raised
                result = cast(ReturnsT, analysis_result.result)
            else:
                try:
                    # this will call prepare().
                    result = super().run(node)
                except Exception as e:
                    analysis_result = AnalysisResult.Error(e)
                    if not typ.do_not_cache:
                        self.passmanager.cache.set(typ, node, analysis_result)
                    raise
                analysis_result = AnalysisResult.Success(result)
                if not typ.do_not_cache:
                    # only set values in the cache for non-proxy analyses.
                    self.passmanager.cache.set(
                        typ, node, analysis_result
                    )
        finally:
            assert analysis_result is not None
            self.passmanager._dispatcher.dispatchEvent(
                AnalysisEnded(typ, node, analysis_result)
            )
        return result

    def apply(self, node: RunsOnT) -> tuple[bool, RunsOnT]: # type: ignore[override]
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
        if node is not self.ctx.module.node:
            # TODO: This limitation might be lifted in the future.
            raise RuntimeError(f'invalid node: {node}, expected {self.ctx.module.node}')
        # TODO: think of a way to cache no-op transformations so that they are not re-run every time an analysis 
        # with transformations in it's dependencies is ran...
        self.passmanager._dispatcher.dispatchEvent(RunningTransform(typ, node))
        try:
            # TODO: Does it actually fixes the new node locations ? I don't think so.
            n = super().run(node)
            # the transformation updated the AST, so analyses may need to be rerun
            if self.update:
                self.passmanager._moduleTransformed(self, self.ctx.module)
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
        if issubclass(self.analysis, __class__): # type: ignore[name-defined]
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
        self.cache = AnalysisCache()
        self._dispatcher: EventDispatcher = passmanager._dispatcher
        self._dispatcher.addEventListener(
            ClearAnalysisEvent, self._onClearAnalysisEvent
        )
        self.__pm = passmanager

    def gather(self, analysis: PassLike, node: AnyNode) -> Any:
        """
        Call an L{analysis} on any node in the system.
        """
        if not TYPE_CHECKING:
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
            a._attach(_RestrictedPassManager(analysis, self), self.__pm._ctx)
            ret = a.run(node)
        
        return ret

    def apply(self, transformation: PassLike, node: Any) -> tuple[bool, AnyNode]:
        """
        Call a C{transformation} on a module C{node}.
        """
        if not TYPE_CHECKING:
            if not issubclass(transformation, Pass):
                raise TypeError(f"unexpected pass type: {transformation}")

        if self.__pm.modules[node] is not self.module:
            return self.__pm.apply(transformation, node)
        
        with self.__pm._ctx.pushPass(transformation, node):
            a = transformation()
            a._attach(_RestrictedPassManager(transformation, self), self.__pm._ctx)
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

class _RestrictedPassManager:
    """
    A proxy to the L{PassManager} instance that makes sure it is used correctly depending on the pass it's attached to.
    
    - Restrict intra-module passes L{gather} and L{apply} so they can't dynamically depend on inter-modules passes.
    - Disallow access to L{PassManager.modules} since it should always be accessed with L{passmanager.modules} analysis.
    - Disallow access to L{PassManager.add_modules} and L{PassManager.remove_module} in the context of an analysis.

    """
    def __init__(self, passs: PassLike, pm: IPassManager) -> None:
        self.__pm = pm
        
        # 1
        if not passs._isInterModuleAnalysis():

            def gather(analysis_: PassLike, node: AnyNode) -> Any:
                if analysis_._isInterModuleAnalysis():
                    raise TypeError(f'You must list {modules.__qualname__} in your pass dependencies to gather this pass: {analysis_}')    
                return pm.gather(analysis_, node)        
            
            def apply(transformation_: PassLike, node: AnyNode):
                if transformation_._isInterModuleAnalysis():
                    raise TypeError(f'{modules.__qualname__} must be in your pass dependencies to apply this pass: {transformation_}')  
                return pm.apply(transformation_, node)        

            self.apply = apply
            self.gather = gather
        
        # 3
        if issubclass(passs, Analysis):
            def add_module(_):
                raise RuntimeError('cannot add a module from within an analysis')
            def remove_module(_):
                raise RuntimeError('cannot remove a module from within an analysis')
            
            self.add_module = add_module
            self.remove_module = remove_module
        
    # 2
    @property
    def modules(self):
        raise RuntimeError(f'You must access the modules with the {modules.__qualname__} dependencies')
    
    # Forward all attribute to self.__pm if it's not defined in this class.
    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            try:
                return getattr(self.__pm, name)
            except AttributeError:
                raise e


# TODO: Reconsider this class!
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



class PassManager:
    """
    Front end to the pass system.
    One L{PassManager} can be used for the analysis of a collection of modules.
    """

    def __init__(self) -> None:
        
        self._dispatcher = d = EventDispatcher()
        self._ast = astsupport = ASTCompat(d)
        _init_support(self)

        self.modules = ModuleCollection(d, astsupport)
        """
        Contains all the modules in the system.
        """

        
        self._passmanagers = pms = _PassManagerCollection(self)
        self._caches = _Caches(pms)
        self._ctx = PassContext(self.modules)

    def add_module(self, mod: Module) -> None:
        """
        Adds a new module to the pass manager.
        Use PassManager.modules to access modules.
        """
        self._dispatcher.dispatchEvent(ModuleAddedEvent(mod))

    def remove_module(self, mod: Module) -> None:
        """
        Remove a module from the passmanager.
        This will allow adding another module with the same name or same module node.
        """
        self._dispatcher.dispatchEvent(ModuleRemovedEvent(mod))

    def gather(self, analysis: PassLike, node: AnyNode) -> Any:
        """
        High-level function to call an L{Analysis} on any node in the system.
        """
        mod = self.modules[node]
        mpm = self._passmanagers[mod]
        return mpm.gather(analysis, node)

    def apply(self, transformation: PassLike, node: ModuleNode) -> tuple[bool, AnyNode]:
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
    
    # # Support gast library
    # if 'gast' in sys.modules:
    #     import gast
    #     # At the moment this strategy provides the same information as the standard library one,
    #     # but that might evolve in the future.
    #     class gast_lib:
    #         iter_child_nodes = standard_lib.iter_child_nodes
        
    #     pm.support(gast_lib)
    
    # # Support parso library
    # if 'parso' in sys.modules:
    #     import parso.tree # type: ignore
    #     class parso_lib:
    #         @staticmethod
    #         def iter_child_nodes(node):
    #             if isinstance(node, parso.tree.BaseNode):
    #                 return node.children
    #             raise NotImplementedError()
        
    #     pm.support(parso_lib)

    # # Support astroid library
    # if 'astroid' in sys.modules:
    #     import astroid # type: ignore
    #     class astroid_lib:
    #         @staticmethod
    #         def iter_child_nodes(node):
    #             if isinstance(node, astroid.NodeNG):
    #                 return node.children
    #             raise NotImplementedError()
        
    #     pm.support(astroid_lib)


class Statistics:
    def __init__(self, dispatcher: EventDispatcher) -> None:
        pass

    def _onRun(self, event):...
    def _onFinish(self, event):...