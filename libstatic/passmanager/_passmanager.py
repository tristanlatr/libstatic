"""
Core of the machinery.
"""
from __future__ import annotations

import abc
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from itertools import product

from typing import (
    Callable,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Mapping,
    NotRequired,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    TypedDict,
    cast,
    overload,
)

import attrs

from libstatic._lib.structures import FrozenDict, Indexer

from ._model import Forest, Tree, Node, Level, PassKind, Element, Node, RootNode
from ._passe import PassInstance, PassKind, PassLike, transformation

if TYPE_CHECKING:
    from libstatic.passmanager._caching import PassPattern

type _ElementPath = (
    tuple[Forest,] | tuple[Forest, Tree] | tuple[Forest, Tree, Node]
)
"""
The path of an element in the system under one of these forms: 
    
    - forest
    - forest, tree
    - forest, tree, node
"""
type _SimpleElementPath = tuple[()] | tuple[Tree,] | tuple[Tree, Node]
"""
What's left from the element path when the forest is trimmed.
"""

type _PassRun = tuple[PassInstance, _ElementPath, Mapping[str, Any]]
"""
A "pass run" more or less encapsulate information 
passed to the PassManager.run method: 

    - pass, 
    - the element path,
    - and the run options
"""


class PassCycleError(RuntimeError):
    """
    Raised when a pass run is requested recursively without a different pass run.

    ``passrun`` is the repeated pass instance, element path, and run options
    tuple.  It is exposed so callers can catch the exception and select a
    different pass or run strategy.
    """

    def __init__(self, passrun: _PassRun) -> None:
        self.passrun = passrun
        super().__init__(f"cycle detected with pass: {passrun}")


class PassRunMeta:
    """
    Encapsulate the data passed arround in between the context and the runner.
    """

    __slots__ = ("used_paths",)

    def __init__(self) -> None:
        self.used_paths: frozenset[_ElementPath] = frozenset()


class PassContext:
    """
    Class that does the book-keeping of the chains of passes runs.

    The context tracks the paths used by each pass and its dependencies.
    """

    __slots__ = ("_metadata_stack",)

    # Maintains a stack of running passes and the paths used by each pass.
    def __init__(self) -> None:
        self._metadata_stack: dict[_PassRun, set[_ElementPath]] = {}

    @property
    def _current_passrun(self) -> _PassRun:
        try:
            return next(reversed(self._metadata_stack))
        except StopIteration:
            raise RuntimeError("no pass is currently running")

    @property
    def current_pass_kind(self) -> PassKind | None:
        if not self._metadata_stack:
            return None
        passe, _, _ = self._current_passrun
        return passe.proto.kind

    @contextmanager
    def _push_pass(self, passrun: _PassRun) -> Iterator[PassRunMeta]:
        passe, epath, options = passrun
        frozen_options: FrozenDict[str, Any] = FrozenDict(options)
        try:
            hash(frozen_options)
        except TypeError as error:
            raise TypeError("pass run options must be hashable") from error
        passrun = (passe, epath, frozen_options)

        if passrun in self._metadata_stack:
            raise PassCycleError(passrun)

        self._metadata_stack[passrun] = {epath}

        # Nested runs contribute their used paths to this pass's metadata.
        # Direct forest reads are represented by a forest-only path.
        meta = PassRunMeta()

        try:
            yield meta
        finally:
            # Without finally statement, if the pass raises, the code after yield never runs.
            
            # This is just in case someone does something stupid
            if __debug__:
                e = next(reversed(self._metadata_stack))
                if e is not passrun:
                    raise RuntimeError(f"pass context is confused: {e} is not {passrun}")
                del e

            # Pop the pass and record all paths used by it and its dependencies.
            used_paths = self._metadata_stack[passrun]
            del self._metadata_stack[passrun]

            meta.used_paths = frozenset(used_paths)

            if self._metadata_stack:
                curr = self._current_passrun
                # propagate the used paths
                self._metadata_stack[curr].update(used_paths)

@attrs.frozen(slots=True)
class UnpreparedPass:
    """
    The object represents a pass not yet prepared to run.
    """
    passe: PassInstance;    pointer: _ElementPath;    
    passmanager: PassManager
    run_options: Mapping[str, Any]

@attrs.frozen(slots=True)
class PreparedPass:
    """
    The object represents a pass ready to run.
    """
    passe: PassInstance;    pointer: _ElementPath;    connector: Connector
    run_options: Mapping[str, Any]

@attrs.frozen(slots=True)
class CompletedPass:
    """
    The object represents a pass that has been completed.
    It is returned from L{PassManager.run} method.
    """

    passe: PassInstance;    pointer: _ElementPath;    connector: Connector
    # Strores the connector so subsequent analyses can be run from inside hooks applied
    # a pass has been run (with the `COMPLETED` kind of hooks).
    # # TODO (phase-one): Rephrase and precise.
    # Attention: great care should be taken not to create situation where 
    # infinite recuresion is possible, i.e. it's safe to call any analyses from within
    # a hook applied to transformations. wait is it ? cannot analyses depend on transformations?
    run_options: Mapping[str, Any]
    meta: PassRunMeta
    #: 
    #: None for transformations.
    result: Any
    #: Always False for analyses.
    update: bool
    
    # Optionals
    attributes: Mapping[str, Any]
    """
    Including complete status, preserved analyses, etc.
    """

@attrs.frozen(slots=True)
class FailedPass:
    """
    The object represents a pass that raised an exception while running.
    """
    passe: PassInstance;    pointer: _ElementPath;    connector: Connector
    run_options: Mapping[str, Any]
    failure: Exception
    

class _PassDependencyDescriptor:
    """
    Simple container for a callback.
    We kinda re-implement part of the descriptor protocol here.

    @see: L{Dependencies.__getattribute__}
    """

    def __init__(self, callback: Callable[[], Any]) -> None:
        self.callback = callback


# TODO: Optimize me with __slots__
class Dependencies:
    """
    Container for dependencies.
    """
    # This class is untypable by nature since it is highly dynamic...
    # the attributes names depend on the listed dependencies, this
    # would require a mypy plugin/custom transformer to be understandable.

    def __getattribute__(self, name: str) -> Any:
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            attr = attr.callback()
            # setattr(self, name, attr) # act like a cached_property?
            # TODO (phase-three): Think of the implications of doing so regarding the caching...
            # might interfere with the invalidation of some analysis in the cache.
        return attr


@attrs.frozen(slots=True)
class Connector:
    """
    Connector to the passmanager, from inside a pass function.
    This is what we get as the first argument of pass functions like::

        @analysis(on=AST)
        def stuff(connector: Connector, node): ...
    """

    deps: Dependencies  #: Namespace containing the declared dependencies
    gather: Callable[..., Any]  #: See L{PassManager.gather}
    apply: Callable[..., bool]  #: See L{PassManager.apply}
    run: Callable[..., CompletedPass]  #: See L{PassManager.run}

# TODO (phase-one): Create a hook that patches or replace the connector to enabled
# a restrictive - testing mode. This will improve performance by not restict anything 
# outside of the tests for isntance: 
# - Disallow transformations from analyses
# - Disallow outer level transformations from transformations (because of the quadratic nature of the operation)
# - Disallow implicitely adding a new (DONE)

type CastableToDict = Iterable[tuple[str, Any]] | dict[str, Any]

class AnalysisReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{analysis} function.
    """
    result: Any
    complete: NotRequired[bool]

class TransformationReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{transformation} function.
    """
    update: bool
    preserve: NotRequired[Iterable[PassLike | PassPattern | str]]


_Trdict = TypeVar("_Trdict", TransformationReturnMap, AnalysisReturnMap)

# typing this framework turns out more difficult that expected...
@attrs.frozen(slots=True)
class Runner(abc.ABC, Generic[_Trdict]):
    """
    The runner and subclasses do the heavy lifting...

    The process look like this:
    -> push 
        -> lookup in cache -> return early if present
        -> prepare the pass
            -> apply dependent transformations
            -> gather analysis results (maybe lazy)
            -> create the PreparedPass instance
        -> apply before hooks 
        -> run the pass
    -> pop
    -> process results 
    -> create the CompletedPass 
    -> apply after hooks 
    -> maintain cache.
    """

    _passmanager: PassManager # TODO (phase-two): Instead of the PassManager, the runner
                              # should receive the required methods/interfaces from
                              # the passmanager. This mean that the unprepared pass instances
                              # can't have a direct reference to the passmanager neither.
    _passe: PassInstance
    _pointer: _ElementPath
    _options: Mapping[str, Any]

    @contextmanager
    def push(self) -> Iterator[PassRunMeta]:
        with self._passmanager._ctx._push_pass(
            (self._passe, self._pointer, self._options)) as meta:
            yield meta

    @staticmethod
    def prepare(unprepared: UnpreparedPass) -> PreparedPass:
        """
        Prepare the pass connector and apply dependent transforms before running a pass.
        """
        # TODO (phase-one): Like in LLVM, we DO impose some restriction about what pass can be run under
        # what context, BUT there is one thing we do not do at the moment: that is to
        # disalow NODE transformations to run outer lever analyses. 
        # It should only be able to access cached results (through the "deps" or gather() cached results). 
        # For the simple reason that 
        # if a NODE transformation is being run on all function in the module, and let's say 
        # it always uses a TREE pass and always updates the function ast without
        # preserving the used analysis result, well... this will result in quadratic 
        # run time like O(number of function * number of function)
        # but maybe the LLVM docs are more clear about this: 
        #   https://llvm.org/docs/NewPassManager.html#using-analyses
        
        # Note that the issue is exactly the same for TREE transformations / FOREST passes. 
        # So basically: all TREE or NODE Transformations MUST preserve all outer scope pass
        # results they might use!!!
        # (I did not say "analyses results they might use", but 
        # "pass result", becasue a transformation results might, in the future, end up stored 
        # in cache as well when it's a no-op - but for now it just means
        # that NODE or TREE Transformation simply cannot depend on outer scope transformations AND
        # MUST preserve all outer scope analysis results they might use!!!.)

        p = unprepared.passe
        passe_proto = p.proto
        element: _SimpleElementPath = unprepared.pointer[1:]
        pm = unprepared.passmanager

        # validate the runtime type if running on nodes
        if passe_proto.runs_on == Level.NODE:
            if not element:
                raise AssertionError
            runs_on_type = passe_proto.runs_on_type
            
            
            if not isinstance(element[-1], runs_on_type):
                # This can happen when defining a pass that only runs on ast.Module 
                # and then calling it from a pass that run on a child node, expecting 
                # the framework to understand that the dependent
                # analysis should run on the enclosing Module. 
                # It doesn't work like that at the moment; since
                # the passmanager doesn't understand the hierarchy in 
                # between ast.Module and, ast.BinOp, let's say.
                # If your pass requires to run on the root of the parse tree, 
                # use on=passmanager.Tree
                # and access the module instance with '.root' attribute.
                raise TypeError(
                    f"unexpected type, got {element[-1]!r}, "
                    f"should be of type {runs_on_type!r}"
                )

        # Apply all transformations eagerly, since we use a descriptor for all analyses results
        # within themself inside analysis, 
        # we need to transitivsely iterate dependent tranforms and apply then now.
        # TODO (phase-one): this might run transitive transformations many times, 
        # so we should really cache the no-op transformation facts...
        
        passe_proto_kind = passe_proto.kind
        for _t in passe_proto.get_all_dependencies(pm.get_passe):
            t_proto = _t.proto

            if t_proto.kind != PassKind.TRANSFORMATION:
                continue
            
            # Analyses Should be allowed to run transformations at all. 
            # We should be able to assume that analyses won't change the analyzed code!
            # At the same time, we should still be able to list transformation as dependencies
            # of analyses, BUT, these transformations dependencies serves only to validate 
            # that they already have been applied to the analyzed code (and a follow-up application
            # would not change the analysed code). This restriction will likely 
            # means that a analysis can only "depend" on a single transformation since once transforantion
            # will invalidate the cached result of any other transformation applied earlier, except if it's 
            # explicitely marked as preserved. This is an OK compromise if it's well documented.
            if passe_proto_kind == PassKind.ANALYSIS:
                raise TypeError(
                    f"{p} is an analysis and cannot depend on transformation {_t}"
                )
            
            if (t_runs_on := t_proto.runs_on) < (p_runs_on := passe_proto.runs_on):
                # We might be able to check the config to see if pass can be run
                # on root nodes or not; but this will increase complexity
                # EDIT: Actually this wouldn't be a good change since different tree
                # implementation might use the root object CompilationUnit several times
                # like in the parsed javalang AST. 
                if t_runs_on == Level.NODE and p_runs_on == Level.TREE:
                    apply_on_element: _SimpleElementPath = element

                else:
                    # it's not clear from the code but that's the only
                    # possible value for this variable at this point.
                    if __debug__:
                        assert p_runs_on == Level.FOREST
                    # This is a limitation, but it's easy to write a simple wrapper
                    # on client side.
                    raise TypeError(
                        f"{p} cannot depend - even transitively - on {_t}. "
                        "A pass can only depend on transformations "
                        "that runs on a compatible or enclosing level."
                    )
            elif t_runs_on > p_runs_on:
                if passe_proto_kind == PassKind.TRANSFORMATION:
                    raise TypeError(
                        # TODO (phase-one): this limitation might be lifted in the future IF we can
                        # ignore the transformation because we KNOW it's not going to
                        # update the content. 
                     "transformations cannot depend on enclosing level transformations")
                        # TODO (phase-one): Enforce this through the connector run() as well.
                
                # the dependency runs on a upper scope level, trim what's required
                lvldiff = t_runs_on - p_runs_on
                apply_on_element = element[:-lvldiff]  # type:ignore[assignment]
            else:
                # same level
                apply_on_element = element

            pm.apply(_t, *apply_on_element)

        # create the analysis dependencies namespace.
        deps = Dependencies()
        for _a in passe_proto.get_dependencies(pm.get_passe):
            a_proto = _a.proto
            if a_proto.kind != PassKind.ANALYSIS:
                continue
            if _missing := _a.missing_param():
                # dependency is missing a required parameter 
                # and cannot be presented as a descriptor.
                # Instead of trying to do something smart and complex, we fail early.

                raise TypeError(
                    f"{p} cannot list {_a} in dependencies because it is missing "
                    f"a required parameter {_missing!r}"
                )

                # So for instance a NODE analysis 'attribute' 
                # which require a 'name' parameter
                # @analysis(on=ast.AST)
                # def my_pass(c, node):
                #   c.gather(attribute('some_name'), 'some_other_module_name')
                #   c.gather(attribute('some_other_name'), 'some_other_module_name', class_def)
            
            cache_only_sub = False
            # TODO (phase-two): More code should be shared with the first for loop up there...
            if (a_runs_on := a_proto.runs_on) < (p_runs_on := passe_proto.runs_on):
                # the dependency runs on a lower scope level
                if a_runs_on == Level.NODE and p_runs_on == Level.TREE:
                    # a node analysis can implicitely be called on a tree, but
                    # other kinds of "promotions" are not supported.
                    dep_element: _SimpleElementPath = element
                else:
                    # this is true because we have only 3 levels of elements.
                    if __debug__:
                        assert p_runs_on == Level.FOREST
                    raise TypeError(
                        f"{p} cannot list {_a} in dependencies. "
                        "A pass can only depend on analyses "
                        "that runs on a compatible or enclosing level."
                    )
            elif a_runs_on > p_runs_on:
                # the dependency runs on a upper scope level, trim what's required
                lvldiff = a_runs_on - p_runs_on
                dep_element = element[:-lvldiff]  # type:ignore[assignment]

                # TODO (phase-one): Enforce this thru the connector run() as well (
                #   in a hook of course).
                cache_only_sub |= passe_proto_kind == PassKind.TRANSFORMATION 
            else:
                # same level
                dep_element = element

            if passe_proto_kind == PassKind.TRANSFORMATION:
                # TODO (phase-one): That's good stuff but we need to move it to the caching.py
                # module inside some hook. How to... -> Simply with a prepared-pass
                # hook that calculates the value of cache_only_sub and call 
                # the connector'deps with object.__getattribute__() to pass 
                # For transformations, the depending analyses MUST be computed eagerly
                # because it could cause undefined behavior if it's lazily 
                # bound like for analyses since the transformation might or 
                # might not affect the tree before/after using an analysis result.
                setattr(deps, a_proto.name, pm.gather(_a, *dep_element, 
                                                      # only pass cache_only if we're 
                                                      # dealing with a transformation.
                                                      cache_only=cache_only_sub))
            else:
                # Convert the dependency to a lazy descriptor. 
                callback: Callable[[], Any] = partial(pm.gather, _a, *dep_element, 
                                                      cache_only=cache_only_sub)
                setattr(deps, a_proto.name, _PassDependencyDescriptor(callback))

        # create the namespace
        connector = Connector(
                deps=deps,
                gather=pm.gather,
                apply=pm.apply,
                run=pm.run,
            )
        
        return PreparedPass(
            passe=unprepared.passe,
            pointer=unprepared.pointer,
            connector=connector, 
            run_options=unprepared.run_options,
        )

    # It's a static method to ensure the prepared pass
    # alone is sufficient to actually run a pass.
    @staticmethod
    def do_pass(prepared: PreparedPass) -> _Trdict:
        # call the pass function
        _res = prepared.passe.proto.do_pass(
            prepared.connector, 
            prepared.pointer[-1], 
            **prepared.passe.args)
        if not isinstance(_res, dict):
            # cast the result to dict if it's not already a dict, 
            # like a generator
            return cast(_Trdict, dict(_res))
        return cast(_Trdict, _res)
    
    @staticmethod
    def _apply_hooks(
        hooks: Hooks, 
        obj: THookObj, 
        when: Trigger,
        ) -> THookObj:
        # TODO (phase-two): Like in requests.Request object, the Pass themselve might want to carry over some
        # hooks, so this method will need to adjust.

        for h in hooks.get(when, obj.passe.proto.kind, obj.passe.proto.runs_on):
            obj = h(obj) or obj
        return obj

    def run(self) -> CompletedPass:
        with self.push() as meta:
            unprepared = UnpreparedPass(self._passe, self._pointer, self._passmanager, 
                                        run_options=self._options)
            unprepared = self._apply_hooks(self._passmanager.hooks, 
                                           unprepared, when=Trigger.UNPREPARED)
            
            # UNPREPARED hooks can either return None, another UnpreparedPass instance or 
            # directly a CompletedPass instance, in which case the instance is returned as-is.
            # This is a special case of the hook logic to accomodate the caching logic. 
            if isinstance(unprepared, CompletedPass):
                return unprepared
            assert isinstance(unprepared, UnpreparedPass), f'hook returned unsupported object type: {unprepared}'
            
            prepared: PreparedPass = self.prepare(unprepared)
            prepared = self._apply_hooks(self._passmanager.hooks, 
                                         prepared, when=Trigger.PREPARED)
            
            try:
                rdict = self.do_pass(prepared)
            except Exception as failure:
                failed = FailedPass(prepared.passe, prepared.pointer, 
                                    prepared.connector, 
                                    run_options=prepared.run_options,
                                    failure=failure)
                maybe_failed = self._apply_hooks(self._passmanager.hooks,
                                                 failed, when=Trigger.FAILED)
                # FAILED hooks can either return None, another FailedPass instance or 
                # a CompletedPass instance, in which case the instance is returned as-is
                # (skipping completed passes hooks!).
                # This is a special case of the hook logic to accomodate error handling. 
                if isinstance(maybe_failed, CompletedPass):
                    return maybe_failed
                assert isinstance(maybe_failed, FailedPass), f'hook returned unsupported object type: {maybe_failed}'
                raise maybe_failed.failure

        result: CompletedPass = self.make_completed_pass(rdict, meta, prepared)
        result = self._apply_hooks(self._passmanager.hooks,
                                   result, when=Trigger.COMPLETED)
        # TODO (phase-one) Move this to caching.py inside an completed pass hook that 
        # increase the revision of the tree that transformation changed somthing if applicable.
        # self.maintain_cache(result)
        return result

    @abc.abstractmethod
    def make_completed_pass(self, rdict: _Trdict, 
                            meta: PassRunMeta,
                            prepared: PreparedPass) -> CompletedPass:
        ...

@attrs.frozen(slots=True)
class AnalysisRunner(Runner[AnalysisReturnMap]):

    def make_completed_pass(self, rdict: AnalysisReturnMap, 
                            meta: PassRunMeta, 
                            prepared: PreparedPass) -> CompletedPass:
        # By default all analyses are complete. 
        rdict.setdefault("complete", True)
        return CompletedPass(
            self._passe,
            self._pointer,
            prepared.connector, 
            run_options=prepared.run_options,
            
            meta=meta,
            result=rdict.pop("result"), # type: ignore
            update=False,
            attributes=rdict
        )

@attrs.frozen(slots=True)
class TransformationRunner(Runner[TransformationReturnMap]):
    
    def make_completed_pass(self, rdict: TransformationReturnMap, 
                            meta: PassRunMeta,
                            prepared: PreparedPass) -> CompletedPass:
        
        # TODO (phase-one): Implement the PreservedAnalyses checks to the caching hook in caching.py
        # preserved = PreservedAnalyses(self._passmanager.get_passe, 
        #                               rdict.get('preserved', []))        
        return CompletedPass(
            self._passe,
            self._pointer,
            prepared.connector, 
            run_options=prepared.run_options,

            meta=meta,
            update=rdict.pop("update"), # type: ignore
            result=None,
            attributes=rdict,
        )


class PassManager:
    """
    Front end to the pass system.
    One L{PassManager} can be used for the analysis of a collection of trees.
    """

    def __init__(self, plugins: Iterable[IPluginFactory] = (),
                # Currently the core of the PassManager 
                # is library agnostic, and should stay that way. 
                ) -> None:

        self.hooks = Hooks()
        self.trees = Forest()
        
        self._preconfigured: dict[str, PassLike] = {}
        self._ctx = PassContext()
        self._install_plugins(plugins)

    def _install_plugins(self, plugins: Iterable[IPluginFactory]) -> None:
        # This method should only be called once at initialization time, 
        # and not after the PassManager has been used.
        # NOTE: The cache is just a plugin like others.
        for i in map(lambda _: _(), plugins):
            for plugin in i.register(self):
                if hasattr(self, name:=plugin.name):
                    raise ValueError(f'The PassManager already has an attribute named {name!r},'
                                        f' please use another name for the plugin {plugin!r}')
                # this makes the PassManager class impossible to type correctly; 
                # an idea to workaround the issue would be to provide type-hint-only mixin classes
                # that would be used under a TYPE_CHECKING block to annotate the PassManager 
                # in use in the code. Of course that works only when we known in advance
                # the list of plugins we're gonna be end up using.
                setattr(self, name, plugin)

    def configure(self, passe: PassLike, name: str) -> None:
        """
        Configure a string alias for a pass. A common use case such feature is to provide
        different kind of the same analyses that are registed under the same name by 
        several plugins. This allows distinction in between the implementation of an
        analyses and the kind of information it computes
        
        :see: `get_passe`
        """
        self._preconfigured[name] = passe._replace(name=name)
    
    def get_passe(self, name: str) -> PassLike:
        """
        Get a pass from it's pre-configured name.

        :see: `configure`
        """
        return self._preconfigured[name]

    def apply(self, transform: PassLike | str, *element: str | Element, **kwargs: Any) -> bool:
        """
        High level method to run a tansformation.
        """
        if isinstance(transform, str):
            transform = self.get_passe(transform)
        if transform.proto.kind != PassKind.TRANSFORMATION:
            raise TypeError
        return self.run(transform, *element, **kwargs).update

    def gather(self, analysis: PassLike | str, *element: str | Element, **kwargs: Any) -> Any:
        """
        High level method to run an analysis.

        # :param cache_only:
        #     Forces the completed pass instance to be fetch from the cache. 
        #     If the object is not found in the cache, it will raise an error.
        """
        if isinstance(analysis, str):
            analysis = self.get_passe(analysis)
        if analysis.proto.kind != PassKind.ANALYSIS:
            raise TypeError
        return self.run(analysis, *element, **kwargs).result

    # One of the design priciples that governs the usage of the run() and friends methods
    # is that for a given node, the passmanger cannot identify which tree
    # this node lives in. This is why in order to run a NODE analysis on anything else than the root
    # node of a tree, users of the library need to pass (at least) the tree identifer as 
    # the second argument AND the actual node instance as the third.

    def run(self, passe: PassLike | str, *element: str | Element, **options:Any) -> CompletedPass:
        """
        Method to run any kind of pass 
        and get a L{CompletedPass} instance in return.

        :param passe: A Pass instance or a pass prototype.
        :param element: The element on which to run the pass.

            - Zero element arguments will run the pass on the entire passmanager forest.
            - First element argument should be the module element, 
              if only one element is given it will run the pass on the specified module. 
              A module can be specified by L{identifier <Tree.identifier>}, 
              L{root <Tree.root>} or by passing the L{Tree} instance directly. 
            - Second element argument should be the node instance, 
              if both module and node elements are given, it will run the pass on 
              the specified node, which should live inside the specified module.
        :param options: Run options passed to the pass. Each option value must be hashable.
        # :param cache_only: Forces the completed pass instance to be fetch from the cache.
        #     If the object is not found in the cache, it will raise an error.
        #     Currently only analyses end up in the cache, so make sure not to use this
        #     with a transformation. 
        """
        if len(element) > 2:
            raise TypeError("this method takes at most 3 positional arguments")
        if isinstance(passe, str):
            passe = self.get_passe(passe)
        if (
            passe.proto.kind is PassKind.TRANSFORMATION
            and self._ctx.current_pass_kind is PassKind.ANALYSIS
        ):
            raise TypeError("analyses cannot run transformations")
        
        # create the pointer
        element = self._prepare_element(element, passe.proto.runs_on)
        pointer: _ElementPath = (self.trees,) + element  # type: ignore
        
        # create the runner
        runner = self._get_runner(passe(), pointer, options)

        # Option names are left unchecked because plugins may intentionally handle
        # otherwise unsupported names, while PassContext validates option hashability.
        return runner.run()

    @overload
    def add(self, tree: RootNode, identifier: str, **attributes: Hashable): ...
    @overload
    def add(self, tree: Tree): ...
    def add(self, tree: Tree | RootNode, 
            identifier: str | None = None, 
            **attributes: Hashable) -> None:
        """
        Add a tree to the passmanager, 
        does nothing if the given tree instance is already in the system.
        """
        if not isinstance(tree, Tree):
            if not identifier:
                raise TypeError('argument "identifier" is required '
                                'if the tree is not an Tree instance')
            tree = Tree(tree, identifier, **attributes)
        elif identifier:
            raise TypeError('argument "identifier" not supported '
                                'when the Tree instance is directly passed')
        elif attributes:
            raise TypeError('keywords not supported '
                                'when the Tree instance is directly passed')
        
        self.apply(_add_tree(tree))

    def remove(self, tree: Tree | RootNode | str) -> None:
        """
        Remove a tree from the passmanager, 
        does nothing if the given tree is not in the system.
        """
        self.apply(_remove_tree(tree))

    def _prepare_element(
        self, element: tuple[str | Element, ...], 
        runs_on: Level, 
    ) -> tuple[Element, ...]:
        len_element = len(element)
        needs_to_append_root = False
        if runs_on == Level.NODE:
            if len_element == 1:
                # Very important for usability!!!
                # a NODE pass can be run on a Tree,
                # in this case use the root module as the node.
                # This is only true if the pass can be run on
                # the root node of the AST.
                needs_to_append_root = True
            elif len_element == 0:
                raise TypeError(
                    "a NODE pass expect at least one "
                    "element argument (up to two), got 0"
                )
        elif runs_on == Level.TREE:
            if len_element != 1:
                raise TypeError(
                    "a TREE pass expect exactly one "
                    f"element argument, got {len_element}"
                )
        elif runs_on == Level.FOREST:
            if len_element != 0:
                raise TypeError(
                    "a FOREST pass expect exactly zero "
                    f"element argument, got {len_element}"
                )

        if element:
            first_element = element[0]
            if not isinstance(first_element, Tree):
                # If the first element is not a tree, try to fetch it from the forest,
                # it can be either a identifier string or the root node of the tree.
                module = self.trees[first_element]
                element = (module,) + element[1:]
            elif first_element not in self.trees:
                # The given tree is not in the system, adding it now could produce inconsistencies
                # when the said tree is added from within an analysis pass (because it should
                # not be permitted to transform the forest from an analysis pass).
                # Instead of trying to do something complex, we simply raise an error and 
                # require the user add the tree before running the pass.
                raise ValueError(f"the given tree {first_element!r} is not in the passmanager, "
                                 "please add it first with PassManager.add()")
            if needs_to_append_root:
                element += (element[0].root,) # type: ignore

        return element

    def _get_runner(self, passe: PassInstance, 
                    pointer: _ElementPath, 
                    options: Mapping[str, Any]) -> Runner:
        map: dict[PassKind, type[Runner]] = {
            PassKind.ANALYSIS: AnalysisRunner,
            PassKind.TRANSFORMATION: TransformationRunner,
        }
        return map[passe.proto.kind](self, passe, pointer, options)


# Internal builtin passes

# These are currently the only forest-level transformations. They change
# membership only, so the cache can advance the forest revision without walking trees.
@transformation(on=Forest, _builtin_forest_transformation=True)
def _add_tree(_: Connector, forest: Forest, *, 
               tree: Tree) -> Iterator[tuple[str, Any]]:
    """
    A forest transformation that adds the given tree.
    """
    if tree in forest:
        yield "update", False
        return
    forest._add(tree)
    yield "update", True


@transformation(on=Forest, _builtin_forest_transformation=True)
def _remove_tree(_: Connector, forest: Forest, *, 
                  tree: Tree | RootNode | str) -> Iterator[tuple[str, Any]]:
    """
    A forest transformation that removes the given tree.
    """
    if tree not in forest:
        yield "update", False
        return
    if not isinstance(tree, Tree):
        tree = forest[tree]
    forest._remove(tree)
    yield "update", True

type _PassState = UnpreparedPass | PreparedPass | CompletedPass | FailedPass
THookObj = TypeVar('THookObj', bound=_PassState)
type Hook[T: _PassState] = Callable[[T], T | None]
"""
A hook is callable that is used to customize the logic 
just before or after a pass runs. 
Note that the hooks won't be called when retreiving 
results from the cache; only when a pass actually runs.

If the hook function returns a value, it'a assumed to replace 
the given object in parameter; this should be of the same type.

If the hook runs at the 'unprepared' step, the object type will be an L{UnpreparedPass}
instance, at 'prepared' the object type will be L{PreparedPass} instance, and,
at 'completed' the object type will be a L{CompletedPass} isntance.

You cannot mutate the object in-place since it's a frozen class. But you can replace
the given object by returning a non-None value.

Use the `Hooks.install()` method to add a new hook. The supported
parameters are: 
    - hook: Hook - the callable
    - when: 'unprepared' or 'prepared' or 'completed'
    - kind: 'analysis' or 'transformation'
    - level:  'node', 'tree' or 'forest'
"""

class IPluginRegistrar(Protocol):
    hooks: Hooks
    def configure(self, passe: PassLike, name: str) -> None: ...
    def get_passe(self, name: str) -> PassLike: ...

class IPluginFactory(Protocol):
    def __call__(self) -> IPlugin: ...

class IPlugin(Protocol):
    """
    An plugin class:
        - has a name
        - can install hooks, which can: 
            - handle gather()/apply()/run() keywords, aka "run options"
            - handle analysis()/transformation() keywords, aka "pass options"
            - handle passe yield points keywords, aka "extra completed pass attributes"
        - can configure a pass alias as string
        - can install other plugins
        
    The PassManager only handled zero-argument callabled returning an instance of IPlugin.
    """
    name: str
    def register(self, r: IPluginRegistrar) -> Iterable[IPlugin]:
        """
        This method MUST at least yield self. 
        Returned instance(s) of the plugin will be stored in the PassManager locals 
        as a attributes matching their `name`. 
        
        A plugin can register other plugins and yield their instances
        from the register() method such that they will be stored in the 
        PassManager locals as well.
        """


class Trigger(IntEnum):
    UNPREPARED = 24
    PREPARED = 23
    COMPLETED = 22
    FAILED = 21


class _HookedAll(IntEnum):
    # for Hooks.install() method
    ALL = 99


def _hooks_indexer() -> Indexer[tuple[Hook[Any], int, int, int]]:
    return Indexer(['hook', 'when', 'kind', 'level'])

@attrs.frozen(slots=True)
class Hooks:
    """
    Container for hooks that customize the pass execution lifecycle.

    Hooks are selected by trigger, pass kind, and execution level. ``install()``
    accepts enum values or case-insensitive names; ``kind='all'`` and
    ``level='all'`` register the hook for every corresponding selector. No
    priority or relative ordering is guaranteed, so hooks should not depend on
    another hook running before or after them.

    At each stage, a hook receives the current pass state. Returning ``None``
    leaves it unchanged; returning a replacement passes that object to the
    next hook. A replacement should be the same state type as the input.
    ``UNPREPARED`` and ``FAILED`` hooks may instead return a ``CompletedPass``
    to short-circuit the run (for example, to serve a cached result or recover
    from a failure). Remaining hooks at that stage still run; a completed
    result returned from either stage skips subsequent execution stages.
    A recovered ``CompletedPass`` from ``FAILED`` bypasses ``COMPLETED`` hooks.

    The pass context records the original run pointer, and the runner is chosen
    from the original pass kind. Hooks should therefore preserve the pass and
    pointer when replacing a state; changing them can make the runner, recorded
    ``used_paths``, result, and cache identity inconsistent. In particular,
    tree and node runs rely on their target pointer being present in
    ``used_paths``.

    Hook states are frozen objects: replace a state by returning a new object
    rather than mutating it. Exceptions raised by hooks propagate; only an
    exception from the pass body is routed through ``FAILED`` hooks. On a
    cache hit, the unprepared stage returns the cached result and later stages
    are skipped.

    >>> my_cb = lambda o: print(o)
    >>> h = Hooks()
    >>> h.install(my_cb, when='completed', kind='analysis', level='tree')
    >>> cbs = list(h.get(Trigger.COMPLETED, PassKind.ANALYSIS, Level.TREE))
    >>> len(cbs)
    1
    >>> cbs[0](2)
    2
    """

    _hooks: Indexer[tuple[Hook[Any], int, int, int]] = attrs.field(factory=_hooks_indexer)

    @staticmethod
    def _cast_string_values(
        when: str | Trigger,
        kind: str | PassKind | _HookedAll,
        level: str | Level | _HookedAll,
    ) -> tuple[Trigger, PassKind | _HookedAll, Level | _HookedAll]:
        # Cast everything to instances of integers.

        def _cast(v: str | int, maps: Iterable[type[IntEnum]]) -> Any:
            if isinstance(v, str):
                v = v.upper()
            if isinstance(v, int):
                return v
            for m in maps:
                try: return m[v]
                except KeyError: continue
            raise KeyError(v)
        
        return (
            _cast(when, [Trigger]), 
            _cast(kind, [PassKind, _HookedAll]),
            _cast(level, [Level, _HookedAll])
        )
        
    def install(self, hook: Hook[Any], *,
        when: str | Trigger,
        kind: str | PassKind | _HookedAll,
        level: str | Level | _HookedAll,
                ) -> None:  
        """
        Add a hook to the system.

        Hooks are selected by trigger, pass kind, and execution level.
        """
        # TODO: Use a priority-based order to apply hooks like we do for pydoctor's post-processing.
        # indtroduce the parameter priority. 

        when, kind, level = self._cast_string_values(when, kind, level)

        kinds = list(PassKind) if kind == _HookedAll.ALL else [kind]
        levels = list(Level) if level == _HookedAll.ALL else [level]
        
        for combo in product(kinds, levels):
            self._hooks.add((hook, when, *combo))

    def get(self, 
        when: Trigger,
        kind: PassKind,
        level: Level,
                ) -> Iterable[Hook[Any]]:
        # This method doesn't support passing string values like install() for performance reason.
        return (k[0] for k in self._hooks.search(when=when, 
                                                 kind=kind, 
                                                 level=level))

    def uninstall(self, hook: Hook[Any]) -> None:
        hooks = self._hooks
        for k in list(hooks.search(hook=hook)):
            hooks.discard(k)
