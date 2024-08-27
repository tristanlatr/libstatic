
# Implementation notes: 
# -> Only true analysis results should be in the cache, proxies do not belong in the cache since they can be re-created on the fly
# -> There is no such things as a analysis that runs on all modules 
#       This will be called client objects/façade and we don't care about them here.
# -> If an analysis raises an exception during processing, this execption should be saved in the cache so it's re-raised
#       whenever the same analysis is requested again.
# -> Name mangling did not work well with dill, so the pattern is to use __{name} instead and NEVER access .__{something}
#       unless throught self.__{something}.

# Future ideas:
# Pass instrumentation: Adding callbacks before and after passes:
# Use this as an extensions system hooked into the dispatcher.
# - First application is to gather statistics about analyses run times
# - Other applications includes generating a tree hash before and after each transform and check it the flag update is well set.

# About caching analysis results in files
# - all modules should have hash keys generated and updated on transformations:
#       1/ local hash (module ast + modulename + is_package, ...): that is only valid for intra-module analysis
#       2/ inter-modules hash: that is a combinaison of the module hash and all it's dependencies'inter-module hashes.

# TODO: Implement isComplete analysis feature that would stay in the cache when a module is added. 
# TODO: Implment transform added / remove nodes in order to optimize visiting time for root module mapping
# TODO: Create an adaptors
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
# TODO: Implement Analysis.like() + add tests
# TODO: Add tests for the Pass.ctx 
# TODO: Think of using multiprocessing to simultaneously run inter-module passes
# on different modules. For the initial implementation to be possible this feature should
# only be accessible as a PassManager factory that would eagerly compute a set of given passes
# on each modules. 
# instance method version: PassManager.parallelPasses(modules, passes) -> None:...
# class method version: PassManager.fromParallelPasses(modules, passes) -> PassManager:...
# The instance method version of the feature would require to re-compute all ancestors of the modules affected by the pass
# The implementation would rely on a new method: PassManager._merge(self, other) that would merge two pm together
# The dispatcher, the cache and the modules also needs a special method to merge them together.


# Ressources:
# LLVM PassManager documentation: https://llvm.org/doxygen/classllvm_1_1PassManager.html
# https://llvm.org/docs/WritingAnLLVMPass.html

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache, partial
from itertools import chain

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Hashable,
    Iterator,
    TypeVar,
    Generic,
    Protocol,
    overload,
)
if TYPE_CHECKING:
    from typing import Self, TypeAlias

from beniget.ordered_set import ordered_set # type: ignore

from .events import (EventDispatcher, ModuleAddedEvent, 
                     ModuleTransformedEvent, ModuleRemovedEvent, AnalysisEnded, 
                     RunningAnalysis, SupportLibraryEvent, TransformEnded, RunningTransform)
from ._modules import Module, ModuleCollection, _Addition, _Removal, SupportsGetItem
from ._astcompat import ASTCompat, ISupport
from ._caching import CacheProxy, AnalysisResult


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
    PassLike: TypeAlias = 'type[Transformation] | type[Analysis]'

ReturnsT = TypeVar("ReturnsT")


class _PassMeta(type):
    # The meta type is only necessary to implement the subclassing with arguments
    # feature of the Pass class.
    # We should not add methods to the meta type unless absolutely required.
    # This hack limits the subclass explosition issue, but not really;
    # by creating cached subclasses when calling the class with arguments.

    # TODO: required parameter should be supported as positional arguments.
    # but we need to handle when there are several values for one argument.
    # TODO: Remove the implicit subclassing with __call__ and only support bind().
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
        return cls.bind(**kwargs)

    # Custom repr so we can showcase the parameter values.
    def __str__(self: type[Pass]) -> str: # type: ignore
        passname = self.__qualname__
        args = []
        for p in self.requiredParameters:
            args.append( f'{p}={getattr(self, p, "<?>")}' )
        for p in self.optionalParameters:
            args.append( f'{p}={getattr(self, p, "<?>")}' )
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


class PassContext:

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
    def pushPass(self, passs: type[Pass], node: AnyNode) -> Iterator[None]:
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


# these object does not exist at runtime it only helps
# for th typing.


class _ICompatibleNodeVisitor(Protocol):
    def visit(self, node:AnyNode) -> Any:...
    @property
    def result(self) -> Any:...


class _ICompatibleNodeTransformer(Protocol):
    def visit(self, node:AnyNode) -> Any:...
    @property
    def update(self) -> bool:...


class IPassManager(Protocol): 
    """
    This interface defines how to run passes.

    @note: This is the passmanager you have access to from inside a L{Pass}, 
        it is stored in the L{Pass.passmanager} instance attribute.
    """
    def apply(self, transformation: type[Pass], node: AnyNode) -> tuple[bool, AnyNode]:...
    def gather(self, analysis: type[Pass], node: AnyNode) -> Any:...

    @property
    def dispatcher(self) -> EventDispatcher:...
    @property
    def cache(self) -> CacheProxy:...
    
    # Private API
    def _getAncestors(self, analysis: ancestors) -> SupportsGetItem[AnyNode, list[AnyNode]]:...
    def _getModules(self, analysis: modules) -> ModuleCollection:...



class IMutablePassManager(IPassManager, Protocol):
    """
    This interface defines how to mutate and access modules in the pass manager system.

    @note: This is the passmanager you get by calling L{PassManager()}.
    """

    def add_module(self, mod: Module) -> None: ...
    def remove_module(self, mod: Module) -> None: ...
    @property
    def modules(self) -> ModuleCollection:...


def _verifyNoJunkParameters(callableName: str, passs: type[Pass], **kwargs:Hashable):
        # verify no junk arguments are slipping throught.
        validNames = ordered_set(chain(passs.requiredParameters, passs.optionalParameters),)
        try:
            junk = next(p for p in kwargs if p not in validNames)
        except StopIteration:
            pass
        else:
            raise TypeError(f"{callableName} does not recognize keyword {junk!r}")


def _verifyParametersOlderValues(passs: type[Pass]):
    # if any of the arguments are not set to their default values (or if a required argument is already set),
    # it means we're facing a case of doudle calling the class: this is not supported and we raise an error.
    # it's supported because it might creates two different classes that do exactly the same thing :/
    _nothing = object()
    if any(
        getattr(passs, promlematic := a, _nothing) is not _nothing
        for a in passs.requiredParameters
    ) or any(
        getattr(passs, promlematic := a, _d) is not _d
        for a, _d in passs.optionalParameters.items()
    ):
        hint = ""
        if passs.__qualname__ != passs.__bases__[0].__qualname__:
            hint = " (hint: analysis subclassing is not supported)"

        raise TypeError(
            f"Specifying parameter {promlematic!r} this way is not supported{hint}, "
            f"you must list all parameters into a single call to class {passs.__bases__[0].__qualname__}"
        )


def _verifyDepNameClash(passs:type[Pass]):
    # verify nothing conflicts with the dependencies names
    _nothing = object()
    try:
        clash = next(
            p
            for p in (d.__name__ for d in passs.dependencies)
            if getattr(passs, p, _nothing) is not _nothing
        )
    except StopIteration:
        pass
    else:
        raise TypeError(
            f"{passs.__qualname__}: invalid class declaration, name {clash!r} is already taken by a dependency"
        )


def _verifyRequiredParamaters(passs: Pass):
    # verify required arguments exists
    _nothing = object()
    try:
        missing = next(
            p
            for p in passs.requiredParameters
            if getattr(passs, p, _nothing) is _nothing
        )
    except StopIteration:
        pass
    else:
        raise TypeError(f"{passs.__class__.__qualname__}() is missing keyword {missing!r}")


class Pass(Generic[RunsOnT, ReturnsT], metaclass=_PassMeta):
    """
    The base class for L{Analysis} and L{Transformation}.
    """
    
    dependencies: tuple[type[Pass], ...] | tuple[()] = () # Sequence[type[Analysis] | type[Transformation]] = ()
    """
    Statically declared dependencies: If your pass requires a previous pass list it here. 
    All L{Transformation}s will be applied eagerly while L{Analysis} result will be wrapped in a descriptor
    waiting to be accessed before running. Must be set at class level. 
    If you require dynamic dependencies, see L{prepareClass}.
    """

    requiredParameters: tuple[str] | tuple[()] = ()
    """
    Some passes looks for specific name that needs to be provided as arguments. 
    Must be statically declared at class level. 
    """

    optionalParameters: dict[str, Hashable] = {}
    """
    Other optional arguments to their default values.
    Must be statically declared at class level. 
    """

    update = False
    """
    Since there is nothing that technically avoids calling L{apply} on an analysis, 
    this flag must be set to L{False} here and potentially overriden in transformations.
    """

    if TYPE_CHECKING:
        def __init__(self) -> None:
            # set in _attach()
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
    def bind(cls, **kwargs:Hashable) -> type[Self]:
        """
        Derive this pass to create a new pass with provided parameters.
        """
        _verifyNoJunkParameters(cls.__qualname__, cls, **kwargs)
        _verifyParametersOlderValues(cls)
        
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
        _verifyDepNameClash(cls)

    @classmethod
    @lru_cache(maxsize=None)
    def _getAllDependencies(cls) -> Collection[type[Pass]]:
        seen = ordered_set()
        def _yieldDeps(c: type[Pass]):
            yield from (d for d in c.dependencies if d not in seen)
            yield from (d for d in chain.from_iterable(
                _yieldDeps(dep) for dep in c.dependencies) if d not in seen)
        seen.update(_yieldDeps(cls))
        return seen
    
    @classmethod
    def isInterModules(cls) -> bool:
        """
        Whether this pass must have knowledge of the other modules in the system.
        """
        # Harmoniozed cost is O(1)
        return modules in cls._getAllDependencies()

    def __getattribute__(self, name):
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            return attr.callback()
        return attr

    def prepare(self, node: RunsOnT):
        """
        Bind analysis result required by this analysis and apply transformations.
        """
        _verifyRequiredParamaters(self)
        
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
                # And instead use self.gather(analysis_name.bind(keyword='stuff'), node)
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

    def _apply(self, node: RunsOnT) -> tuple[bool, ReturnsT]:
        """
        Apply transformation and return if an update happened.
        If self is an analysis, print results.
        """
        raise NotImplementedError(self._apply)

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

    doNotCache = False
    """
    Whether the result of this analysis should never be cached.
    This can be set at class level only.
    """

    isComplete = False
    """
    Whether the result of the analysis is completely full. Only applicable to inter-modules analyses (intra-modules analyses are assumed to alwasy be complete).
    Meaning it will be kept in cache when a new module is added to the system. 
    This is an optimization you don't want to forget about when writting an inter-module analysis.
    This should be set at instance level. Typically an import graph is never complete, but type inference can be complete.
    """

    def run(self, node: RunsOnT) -> ReturnsT:
        typ = type(self)
        self.passmanager.dispatcher.dispatchEvent(
            RunningAnalysis(typ, node)
        )
        try:
            analysis_result = self.passmanager.cache.get(typ, node)
            if analysis_result is not None:  # the result is cached
                # will rase an error if the initial analysis raised
                result = analysis_result.result # type: ignore
            else:
                try:
                    # this will call prepare().
                    result = super().run(node)
                except Exception as e:
                    analysis_result = AnalysisResult.Error(e)
                    if not typ.doNotCache:
                        self.passmanager.cache.set(typ, node, analysis_result, 
                                                   isComplete=self.isComplete)
                    raise
                analysis_result = AnalysisResult.Success(result)
                if not typ.doNotCache:
                    # only set values in the cache for non-proxy analyses.
                    self.passmanager.cache.set(
                        typ, node, analysis_result, 
                        isComplete=self.isComplete
                    )
        finally:
            assert analysis_result is not None
            self.passmanager.dispatcher.dispatchEvent(
                AnalysisEnded(typ, node, analysis_result)
            )
        return result # type: ignore

    def _apply(self, node: RunsOnT) -> tuple[bool, RunsOnT]: # type: ignore[override]
        print(self.run(node))
        return False, node
    
    @classmethod
    def proxy(cls) -> type[NodeAnalysis[GetProxy]]:
        """
        Derive this analysis to return a simple proxy that provide a C{get} method which trigers
        the original analysis on the given node. Designed to be used when declaring L{Pass.dependencies}.
        
        This can be used to avoid calling repetitively C{self.passmanager.gather(analysis, ...)}.

        Proxy analysis are never cached, so they don't need to be accounted for in  L{Transformation.preservesAnalyses}.
        """
        return _AnalysisProxy.bind(analysis=cls)
    
    @classmethod
    def fromNodeVisitor(cls, visitor: type[_ICompatibleNodeVisitor]) -> type[NodeAnalysis]:
        return _AnalysisAdaptor.bind(visitor=visitor)

    @classmethod
    def fromCallable(cls, cb: Callable[[AnyNode], object]) -> type[NodeAnalysis]:
        raise NotImplementedError()
    
    def like(cls, **kwargs: Callable[[object], bool]) -> LikeAnalysisPattern:
        """
        Create a pattern representing several possible derivations of the analysis to be matched against a concrete analysis. 
        Designed to be used when declaring L{Transformation.preservesAnalyses}.

        When creating a "like" pattern, all optional parameters must be given. 
        Required parameters, onther other hand, don't need to be specified and will match any value unless 
        the parameter is explicit given. 

        >>> class has_optional_parameters(NodeAnalysis):
        ...     # We can create an infinity of subclasses of this type since mult can be any ints
        ...     optionalParameters = dict(filterkilled=True, mult=1)
        >>> class my_transform(Transformation):
        ...     # This will only preserves  versions of the analysis with filterkilled=False and mult>0
        ...     preservesAnalyses = (has_optional_parameters.like(filterkilled=lambda v:not v, 
        ...                                                   mult=lambda v: v>0), )

        @param kwargs: The analysis parameters names to the match function. A match function is a one-argument
            callable that returne whether the value for the parameter matches.
        """
        return LikeAnalysisPattern(cls, **kwargs)


class ModuleAnalysis(Analysis[ModuleNode, ReturnsT]):
    """An analysis that operates on a whole module."""

class NodeAnalysis(Analysis[AnyNode, ReturnsT]):
    """An analysis that operates on any node."""

class FunctionAnalysis(Analysis[AnyNode, ReturnsT]):
    """An analysis that operates on a function."""


class ClassAnalysis(Analysis[AnyNode, ReturnsT]):
    """An analysis that operates on a class."""


class LikeAnalysisPattern:
    """
    I represent several derivations of the same analaysis.
    """
    
    def __init__(self, analysis: type[Analysis], 
                 **kwargs: Callable[[object], bool]) -> None:
        
        # Verify if all the optional parameters are given... 
        callableName = f'{analysis.__qualname__}.like'
        _verifyNoJunkParameters(callableName, analysis, **kwargs)
        if not all(problematic:=i in kwargs for i in analysis.optionalParameters):
            raise TypeError(f'{callableName}() is missing keyword {problematic!r}')
        self.__match = kwargs
    
    def matches(self, other: Any) -> bool:
        """
        Whether the given analysis type matches the pattern.

        @note: This is the same as calling C{__eq__}.
        """
        if not isinstance(other, type) or not issubclass(other, Analysis):
            return False
        for k, cb in self.__match.items():
            v = getattr(other, k)
            if not cb(v): 
                break
        else: 
            return True
        return False

    __eq__ = matches


# This would be called a immutable pass in the LLVM jargo.
class modules(Analysis[object, ModuleCollection]):
    """
    Special analysis that results in the mapping of modules: L{ModuleCollection}.
    Use this to access other modules in the system.
    """

    doNotCache = True

    def doPass(self, _: object) -> ModuleCollection:
        return self.passmanager._getModules(self)


# This would be called a immutable pass in the LLVM jargo.
class ancestors(ModuleAnalysis[SupportsGetItem[AnyNode, list[AnyNode]]]):
    """
    Special analysis that results in the mapping of ancestors: L{AncestorsMap}.
    Use this to access ancestors data of any node in the system.
    """
    doNotCache = True

    def doPass(self, _: object) -> SupportsGetItem[AnyNode, list[AnyNode]]:
        return self.passmanager._getAncestors(self)
    

class Transformation(Pass[ModuleNode, ModuleNode]):
    """
    A pass that updates the module's content.
    
    A transformation must never update other modules, but otherwise can change anything in
    the current module including global varibles functions and classes.
    """

    preservesAnalyses: Collection[type[Analysis] | LikeAnalysisPattern] = ()
    """
    One of the jobs of the PassManager is to optimize how and when analyses are run. 
    In particular, it attempts to avoid recomputing data unless it needs to. 
    For this reason, passes are allowed to declare that they preserve (i.e., they don’t invalidate) an 
    existing analysis if it’s available. For example, a simple constant folding pass would not modify the 
    CFG, so it can’t possibly affect the results of dominator analysis. 
    
    By default, all passes are assumed to invalidate all others in the current module as
    well as all other analyses that transitively uses the L{modules} analisys.

    This variable should be overridden by subclasses to provide specific list.
    Can be set at class and/or instance level. 
    """

    
    def __init__(self):
        self.update = False
        """
        It should be True if the module was modified by the transformation and False otherwise.
        """

        # optimization, use recAddNode and recRemoveNode from your transformation
        self._updates: list[_Addition | _Removal] = []
    
    @classmethod
    def prepareClass(cls):
        cls.dependencies += (ancestors, )
        super().prepareClass()

        
    def recAddNode(self, node:AnyNode, ancestor:AnyNode):
        """
        Record that a new node has been added to the tree, this should be called 
        everytime a node is added to properly optimize a transformation.  

        @param node: The new node
        @param ancestor: The parent of the new node, this node must be already
            present in the tree. 
        """
        self.update = True
        self._updates.append(_Addition(node, ancestor))
        
    def recRemoveNode(self, node:AnyNode):
        """
        Record that a node has been removed from the tree, this should be called 
        everytime a node is removed to properly optimize a transformation.  
        
        @param node: The removed node
        """
        self.update = True
        self._updates.append(_Removal(node))
    
    def recReplaceNode(self, oldNode: AnyNode, newNode: AnyNode):
        """
        Record that a node has been replaced, this should be called 
        everytime a node is replaced to properly optimize a transformation.  
        
        @param node: The replaces node
        """
        parent = self.ancestors[oldNode][-1] # node not in the system :/ 
        # this line could raise KeyError or IndexError but
        # it's not caught for performance reasons

        self.update = True
        self._updates.append(_Addition(newNode, parent))
        self._updates.append(_Removal(oldNode))


    def run(self, node: ModuleNode) -> ModuleNode:
        typ = type(self)
        if node is not self.ctx.module.node:
            # TODO: This limitation might be lifted in the future.
            raise RuntimeError(f'invalid node: {node}, expected {self.ctx.module.node}')
        # TODO: think of a way to cache no-op transformations so that they are not re-run every time an analysis 
        # with transformations in it's dependencies is ran...
        self.passmanager.dispatcher.dispatchEvent(RunningTransform(typ, node))
        try:
            # TODO: Does it actually fixes the new node locations ? I don't think so.
            n = super().run(node)
            # the transformation updated the AST, so analyses may need to be rerun
            if self.update:
                self.passmanager.dispatcher.dispatchEvent(ModuleTransformedEvent(self, self.ctx.module))
        finally:
            self.passmanager.dispatcher.dispatchEvent(TransformEnded(self, node))

        return n

    def _apply(self, node: ModuleNode) -> tuple[bool, ModuleNode]:
        new_node = self.run(node)
        return self.update, new_node

    @classmethod
    def fromNodeTransformer(cls, transformer) -> type[Transformation]:
        return _TransformationAdaptor.bind(transformer=transformer)


class GetProxy(Generic[RunsOnT, ReturnsT]):
    """
    Provide L{get} method that defers to an underlying callable.
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


class _AnalysisProxy(NodeAnalysis[GetProxy]):
    """
    An analysis that returns a simple proxy that provide a C{get} method which trigers
    the underlying analysis on the given node.
    """
    
    doNotCache = True

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


class _AnalysisAdaptor(NodeAnalysis[Any]):
    """
    Create an analysis from a compatible node visitor.
    """
    requiredParameters = ('visitor',)

    if TYPE_CHECKING:
        visitor: type[_ICompatibleNodeVisitor]

    def __init_subclass__(cls, **kwargs: Hashable):
        super().__init_subclass__(**kwargs)
        # Override the __name__ so the pass has the visitor name
        cls.__name__ = cls.visitor.__name__

    def doPass(self, node: AnyNode) -> Any:
        vis = self.visitor()
        vis.visit(node)
        return vis.result

class _TransformationAdaptor(Transformation):
    """
    Create a transformation from a compatible node transform.
    """
    requiredParameters = ('transformer',)

    if TYPE_CHECKING:
        transformer: type[_ICompatibleNodeTransformer]

    def __init_subclass__(cls, **kwargs: Hashable):
        super().__init_subclass__(**kwargs)
        # Override the __name__ so the pass has the visitor name
        cls.__name__ = cls.transformer.__name__

    def doPass(self, node: AnyNode) -> Any:
        vis = self.transformer()
        newNode = vis.visit(node)
        self.update = vis.update
        if newNode is not node:
            raise RuntimeError('Transformers must not replace the node passed to the run() method')
        return newNode

class _RestrictedPassManager:
    """
    A proxy to the L{PassManager} instance that makes sure it is used correctly depending on the pass it's attached to.
    
    It implements L{IPassManager}.
    
    - Restrict intra-module passes L{gather} and L{apply} so they can't dynamically depend on inter-modules passes.
    - Disallow access to L{PassManager.modules} since it should always be accessed with L{passmanager.modules} analysis.
    - Disallow access to L{PassManager.add_modules} and L{PassManager.remove_module} in the context of an analysis.

    """
    def __init__(self, passs: type[Pass], pm: IMutablePassManager) -> None:
        self.__pm = pm
        
        # 1
        if not passs.isInterModules():

            def gather(a: type[Pass], node: AnyNode) -> Any:
                if a.isInterModules():
                    raise TypeError(f'{modules.__qualname__} must be in your pass dependencies to run this pass: {a}')  
                return pm.gather(a, node)        
            
            def apply(t: type[Pass], node: AnyNode):
                if t.isInterModules():
                    raise TypeError(f'{modules.__qualname__} must be in your pass dependencies to run this pass: {t}')  
                return pm.apply(t, node)        

            self.apply = apply # type: ignore
            self.gather = gather # type: ignore
        
        # 3
        if issubclass(passs, Analysis):
            def add_module(_):
                raise RuntimeError('cannot add a module from within an analysis')
            def remove_module(_):
                raise RuntimeError('cannot remove a module from within an analysis')
            
            self.add_module = add_module # type: ignore
            self.remove_module = remove_module # type: ignore
        
    # 2
    @property
    def modules(self):
        raise RuntimeError(f'You must access the modules with the {modules.__qualname__} dependencies')
    
    # Default versions only forwards calls: explicit is better than implicit.
    # so we don't rely on __getattribute__(). 
    
    # Public API
    gather = lambda self,a,n: self.__pm.gather(a,n)
    apply = lambda self,t,n: self.__pm.apply(t,n)
    add_module = lambda self,m: self.__pm.add_module(m)
    remove_module = lambda self,m: self.__pm.remove_module(m)
    cache: CacheProxy = property(lambda self: self.__pm.cache) # type: ignore
    dispatcher: EventDispatcher = property(lambda self: self.__pm.dispatcher) # type: ignore
    
    # Private API
    _getAncestors = lambda self,p: self.__pm._getAncestors(p)
    _getModules = lambda self,p: self.__pm._getModules(p)
    
    #TODO: lambda functions are great for simplicity, but not for clarty, documentation and typing...


def _initAstSupport(pm: PassManager):
    """Supports the standard library by defaut"""
    import ast
    pm.support(ast) # type: ignore


class PassManager:
    """
    Front end to the pass system.
    One L{PassManager} can be used for the analysis of a collection of modules.
    """

    def __init__(self) -> None:
        
        self.dispatcher = d = EventDispatcher()
        self._astcompat = astcompat = ASTCompat(d)
        _initAstSupport(self)

        self.modules = ModuleCollection(d, astcompat)
        """
        Contains all the modules in the system.
        """

        self.cache = CacheProxy(self.modules, d)
        self._ctx = PassContext(self.modules)


    def add_module(self, mod: Module) -> None:
        """
        Adds a new module to the pass manager.
        Use PassManager.modules to access modules.
        """
        self.dispatcher.dispatchEvent(ModuleAddedEvent(mod))

    def remove_module(self, mod: Module) -> None:
        """
        Remove a module from the passmanager.
        This will allow adding another module with the same name or same module node.
        """
        self.dispatcher.dispatchEvent(ModuleRemovedEvent(mod))

    def gather(self, analysis: type[Pass], node: AnyNode) -> Any:
        """
        High-level function to call an L{Analysis} on any node in the system.
        """
        if not issubclass(analysis, Analysis):
            raise TypeError(f"unexpected analysis type: {analysis}")

        with self._ctx.pushPass(analysis, node):
            a = analysis()
            a._attach(_RestrictedPassManager(analysis, self), self._ctx)
            ret = a.run(node)
        
        return ret

    def apply(self, transformation: type[Pass], node: ModuleNode) -> tuple[bool, AnyNode]:
        """
        High-level function to call a L{Transformation} on a C{node}.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        """
        if not issubclass(transformation, Pass):
            raise TypeError(f"unexpected pass type: {transformation}")

        with self._ctx.pushPass(transformation, node):
            a = transformation()
            a._attach(_RestrictedPassManager(transformation, self), self._ctx)
            ret = a._apply(node)
        
        return ret

    def support(self, lib: ISupport) -> None:
        """
        Change the support for AST parser. This should be called first if you use the pass manager with 
        a AST parser library that is not the standard library. 

        Only one type of tree can be supported at a time.
        """
        self.dispatcher.dispatchEvent(SupportLibraryEvent(lib))
    
    def _getModules(self, analysis: modules) -> ModuleCollection:
        # access modules from within a pass context.
        if not isinstance(analysis, modules): raise RuntimeError()
        return self.modules

    def _getAncestors(self, analysis: ancestors) -> SupportsGetItem[AnyNode, list[AnyNode]]:
        # access ancestors from within a pass context.
        if not isinstance(analysis, ancestors): raise RuntimeError()
        return self.modules.ancestors

    # def _merge(self, other: PassManager) -> None:
    #     """
    #     Merge the given pass manager into this one.
    #     """
    #     self.cache._merge(other.cache)
    #     self.modules._merge(other.modules)
   

class Statistics:
    def __init__(self, dispatcher: EventDispatcher) -> None:
        pass

    def _onRun(self, event):...
    def _onFinish(self, event):...