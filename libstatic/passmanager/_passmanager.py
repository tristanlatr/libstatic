"""
Core of the machinery.
"""
from __future__ import annotations

import abc
from collections import deque
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from inspect import signature, Parameter
from itertools import chain, product
import weakref

from typing import (
    Callable,
    Collection,
    Container,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    TypedDict,
    overload,
)

import attrs

from libstatic._lib.structures import (
    Cache,
    Indexer,
    FrozenDict,
    OrderedSet,
)



from ._model import Forest, Tree, Node, Level, PassKind, Element, Node, RootNode
from ._passe import PassInstance, PassPrototype, PassKind, PassLike, transformation


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

type _PassRun = tuple[PassInstance, _ElementPath]
"""
A "pass run" stores a pass and on which element it has been run.
"""

class _PassRunMetadata:
    """
    Encapsulate the data passed arround in between the context and the runner.

    Currently this only includes the knowledge level of the pass.
    """

    __slots__ = ("knowledge", "used_paths")

    def __init__(self) -> None:
        self.knowledge: int = None
        self.used_paths: set[_ElementPath] = None


class PassContext:
    """
    Class that does the book-keeping of the chains of passes runs.

    The context tracks which pass requires what "knowledge".
    This concept is closely related to the "pass level" concept in the sens that
    the knowledge can only be one of these three values: forest, tree or node.
    """

    __slots__ = ("_knowledge_stack",)

    # maintains a "stack" of running passes and on which element
    # the stack is implemented as dict because it stores the knowledge
    # level of the pass run as well.
    def __init__(self) -> None:
        self._metadata_stack: dict[_PassRun, tuple[int, set[_ElementPath]]] = {}

    @property
    def _current_passrun(self) -> _PassRun:
        try:
            return next(reversed(self._metadata_stack))
        except StopIteration:
            raise RuntimeError("no pass is currently running")

    @contextmanager
    def _push_pass(
        self, passe: PassInstance, pointer: _ElementPath
    ) -> Iterator[_PassRunMetadata]:

        key: _PassRun = (passe, pointer)

        if key in self._metadata_stack:
            # TODO: Use a exception subclass in order to potentially catch and
            # use another pass instead to break the cycle.
            raise RuntimeError(f"cycle detected with pass: {key}")

            # cast it to int to internalize the int,
            # TODO: is this necessary?
        self._metadata_stack[key] = (int(passe.proto.runs_on), set([pointer]))

        # TODO: Might be interesting to optimize the transformations:
        # Yield a context tracker that is able to say which knowledge
        # the pass accessed as well as the complete list of dependent trees
        # in the case of a forest knowledge analysis.
        # We can do this safely only if no direct access to the forest is done
        # from within the pass. We can do this  by calling gather with a different 
        # tree identifier but the pass should never directly read the content of the Forest.
        meta = _PassRunMetadata()

        # enter context managed code
        yield meta

        # This is just in case someone does something stupid
        if __debug__:
            e = next(reversed(self._metadata_stack))
            if e is not key:
                raise RuntimeError(f"pass context is confused: {e} is not {key}")
            del e

        # pop element from "stack" and record knowledge level in 'meta'.
        pass_run_metadata = self._metadata_stack[key]
        del self._metadata_stack[key]

        meta.knowledge, meta.used_paths = pass_run_metadata[0], frozenset(pass_run_metadata[1])
        # so at this point pass_run_metadata[0] contains the maximum 
        # runs_on level of the "passe"
        # and all it's **ran** dependencies.

        # propagate the knowledge of the dependency
        # pass towards the calling pass if any.
        if self._metadata_stack:
            curr = self._current_passrun
            if self._metadata_stack[curr][0] < pass_run_metadata[0]:
                self._metadata_stack[curr][0] = pass_run_metadata[0]
            
            # propagate the used pointers
            self._metadata_stack[curr][1].update(pass_run_metadata[1])

@attrs.frozen(slots=True)
class UnpreparedPass:
    """
    The object represents a pass not yet prepared to run.
    """
    passe: PassInstance
    pointer: _ElementPath
    passmanager: PassManager
    run_keywords: Mapping[str, Any]

@attrs.frozen(slots=True)
class PreparedPass:
    """
    The object represents a pass ready to run.
    """

    passe: PassInstance
    pointer: _ElementPath
    connector: Connector
    run_keywords: Mapping[str, Any]

@attrs.frozen(slots=True)
class CompletedPass:
    """
    The object represents a pass that has been completed.
    It is returned from L{PassManager.run} method.
    """

    passe: PassInstance
    connector: Connector
    """
    Strores the connector so subsequent analyses can be run from inside hooks applied
    a pass has been run (with the `COMPLETED` kind of hooks).
    
    Attention: great care should be taken not to create situation where 
    infinite recuresion is possible, i.e. it's safe to call any analyses from within
    a hook applied to transformations. wait is it ? cannot analyses depend on transformations?
    """

    # TODO: The whole 'pointer' concept is redundant and unclear.
    # we should just explicitely mention three attributes:
    # - forest, tree and node.
    pointer: _ElementPath
    result: Any
    completeness: bool
    update: bool
    preserved: Container[PassInstance]
    knowledge: int
    run_keywords: Mapping[str, Any]

@attrs.frozen(slots=True)
class FailedPass:
    """
    The object represents a pass that raised an exception while running.
    """

    passe: PassInstance
    pointer: _ElementPath
    connector: Connector
    failure: Exception
    run_keywords: Mapping[str, Any]

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
    # the attributes names depend en the listed dependencies, this
    # would require a mypy plugin/custom transformer to be understandable.

    def __getattribute__(self, name: str) -> Any:
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            attr = attr.callback()
            # setattr(self, name, attr) # act like a cached_property?
                                      # TODO: Think of the implications of doing so
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

    # def gather():...
        # TODO: Verify that the mentioned tree is the current one, othwerwise
        # make this pass forest-wide. 
    
    # TODO: Disallow transformations from analyses
    # TODO: Disallow outer level transformations from transformations
    # TODO:
    # TODO: Disallow implicitely adding a new tree 
    #   from within a non-forest-wide-transfortmation.

type CastableToDict = Iterable[tuple[str, Any]] | dict[str, Any]

class AnalysisReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{analysis} function.
    """
    result: Any

class TransformationReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{transformation} function.
    """
    update: bool


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

    _passmanager: PassManager # TODO: Instead of the PassManager, the runner
                              # should receive the methods run() and push() 
                              # as well as the cache, but nothing more.
    _passe: PassInstance
    _pointer: _ElementPath

    @contextmanager
    def push(self) -> Iterator[_PassRunMetadata]:
        with self._passmanager._ctx._push_pass(self._passe, self._pointer) as meta:
            yield meta

    @staticmethod
    def prepare(unprepared: UnpreparedPass) -> PreparedPass:
        """
        Prepare the pass connector and apply dependent transforms before running a pass.
        """
        # TODO: Like in LLVM, we DO impose some restriction about what pass can be run under
        # what context, BUT there is one thing we do not do at the moment: that is to
        # disalow NODE transformations to run outer lever analyses. 
        # It should only be able to access cached results though the "deps" (or gather() cached results). 
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
        # that NODE or TREE Transformation simply cannot depend on outer scope transformations.)

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
        # within themseft inside analysis, 
        # we need to transitivsely iterate dependent tranforms and apply then now.
        # TODO: this will run transitive transformations many times, 
        # so we should really cache the no-op transformation facts...
        
        passe_proto_kind = passe_proto.kind
        for _t in passe_proto.get_all_dependencies(pm.get_passe):
            t_proto = _t.proto

            if t_proto.kind != PassKind.TRANSFORMATION:
                continue

            # TODO: Should analyses be allowed to depend on transformations at all? NO!
            # We should be able to assume that analyses won't change the analyzed code!
            # At the same time, we should still be able to list transformation as dependencies
            # of analyses, BUT, these transformations dependencies serves only to validate 
            # that they already have been applied to the analyzed code (and a follow-up application
            # would not change the analysed code). If implemented, this restriction will likely 
            # mean that a analysis can only "depend" on a single transformation since once transforantion
            # will invalidate the cached result of any other transformation applied earlier, if not 
            # explicitely marked as preserved. This is probably a ok compromise if it's well documented.
            
            if (t_runs_on := t_proto.runs_on) < (p_runs_on := passe_proto.runs_on):
                # Since a NODE pass can be run on a Tree, we should explicitely
                # accept if we hit this case. BUT what if the transformation only
                # applies to FunctionDef for instance ? well.. then it will fail
                # at the apply() stage with a TypeError.
                # TODO: we might be able to check the config to see if pass can be run
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
                        # TODO: this limitation might be lifted in the future IF we can
                        # ignore the transformation because we KNOW it's not going to
                        # update the content. 
                     "transformations cannot depend on enclosing level transformations")
                    # TODO: Enforce this though the connector run() as well.
                
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
            # TODO: More code should be shared with the first for loop up there...
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

                # TODO: Enforce this thru the connector run() as well.
                cache_only_sub |= passe_proto_kind == PassKind.TRANSFORMATION 
            else:
                # same level
                dep_element = element

            if passe_proto_kind == PassKind.TRANSFORMATION:
                # For transformations, the analyses MUST be computed eagerly
                # because it could cause undefined behavior if it's lazily 
                # bound like for analyses since the transformation might or 
                # might not affect the tree before/after using an analysis result.
                setattr(deps, a_proto.name, pm.gather(_a, *dep_element, 
                                                      # only pass cache_only if we're 
                                                      # dealing with a transformation.
                                                      cache_only=cache_only_sub))
            else:
                # the dependency can be converted to a descriptor
                # TODO: I'm sure there is a faster way to do it
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
            return dict(_res)
        return _res
    
    @staticmethod
    def _apply_hooks(hooks: Hooks, 
        obj: THookObj, 
        when: Trigger,
        knowledge: Level | _HookedNotApplicable,
        ) -> THookObj:
        # TODO: Like in requests.Request object, the Pass themselve might want to carry over some
        # hooks, so this method will need to adjust.

        for h in hooks.get(when, obj.passe.proto.kind, obj.passe.proto.runs_on, knowledge):
            obj = h(obj) or obj
        return obj

    def run(self, **keywords) -> CompletedPass:
        """
        :param cache_only: Only use the cache. Do not actually run anything.
        """
        # TODO: Validate options

        with self.push() as meta:
            unprepared = UnpreparedPass(self._passe, self._pointer, self._passmanager, 
                                        run_keywords=keywords)
            unprepared = self._apply_hooks(unprepared, when=Trigger.UNPREPARED, 
                                       knowledge=_HookedNotApplicable.NA)
            
            # UNPREPARED hooks can either return None, another UnpreparedPass instance or 
            # directly a CompletedPass instance, in which case the instance is returned as-is.
            # This is a special case of the hook logic to accomodate the caching logic. 
            if isinstance(unprepared, CompletedPass):
                return unprepared
            assert isinstance(unprepared, UnpreparedPass), f'hook returned unsupported object type: {unprepared}'
            
            # if completed:=self.from_cache():
            #     return completed
            # elif cache_only:
            #     raise ValueError(
            #         f'result for {(self._passe, self._pointer)!r} not found in cache and cache_only=True')

            prepared: PreparedPass = self.prepare(unprepared) #TODO: should be unprepared.prepare()
            prepared = self._apply_hooks(prepared, when=Trigger.PREPARED, 
                                       knowledge=_HookedNotApplicable.NA)
            
            try:
                rdict = self.do_pass(prepared)  # TODO: should be prepared.do_pass()
            except Exception as failure:
                failed = FailedPass(prepared.passe, prepared.pointer, 
                                    prepared.connector, failure, 
                                    run_keywords=prepared.run_keywords)
                maybe_failed = self._apply_hooks(failed, when=Trigger.FAILED,
                    knowledge=_HookedNotApplicable.NA)
                # FAILED hooks can either return None, another FailedPass instance or 
                # a CompletedPass instance, in which case the instance is returned as-is.
                # This is a special case of the hook logic to accomodate error handling. 
                if isinstance(maybe_failed, CompletedPass):
                    return maybe_failed
                assert isinstance(maybe_failed, FailedPass), f'hook returned unsupported object type: {maybe_failed}'
                raise maybe_failed.failure

        result: CompletedPass = self.make_completed_pass(rdict, meta)
        result = self._apply_hooks(result, when=Trigger.COMPLETED, 
                                   knowledge=Level(result.knowledge))
        # self.maintain_cache(result)
        return result

    @abc.abstractmethod
    def make_completed_pass(self, rdict: _Trdict, meta: _PassRunMetadata) -> CompletedPass:
        ...

@attrs.frozen(slots=True)
class AnalysisRunner(Runner[AnalysisReturnMap]):


    
    def make_completed_pass(self, rdict: _Trdict, meta: _PassRunMetadata) -> CompletedPass:
        # by default all forest knowledge analyses are incomplete and other are complete.
        # TODO: We currently do not validate if a tree or 
        #   node analysis is ever marked as incomplete.
        #   in which case that would be an error of the developers.
        knowledge = meta.knowledge
        rdict.setdefault("completeness", knowledge != Level.FOREST)
        return CompletedPass(
            self._passe,
            self._pointer,
            result=rdict["result"],
            completeness=rdict["completeness"],
            update=False,
            knowledge=knowledge,
            preserved=(),
        )



@attrs.frozen(slots=True)
class TransformationRunner(Runner[TransformationReturnMap]):
    

    def make_completed_pass(self, rdict: _Trdict, meta: _PassRunMetadata) -> CompletedPass:

        preserved = PreservedAnalyses(self._passmanager.get_passe, 
                                      rdict.get('preserved', []))

        pointer = self._pointer
        passe = self._passe
        runs_on = passe.proto.runs_on
        knowledge = meta.knowledge
        
        return CompletedPass(
            passe,
            pointer,
            update=rdict["update"],
            preserved=preserved,
            completeness=False,
            result=None,
            knowledge=knowledge,
        )


class PassManager:
    """
    Front end to the pass system.
    One L{PassManager} can be used for the analysis of a collection of trees.
    """

    def __init__(self, config: Config = default_config, 
                # Currently the core of the PassManager 
                # is library agnostic, and should stay that way.
                 ) -> None:
        self.config = config
        self.hooks = Hooks()
        self.trees = Forest()
        
        self._preconfigured: dict[str, PassLike] = {}
        
        # The cache should be just an instrumentation like others.
        # self.cache = PassManagerCache(Cache(CACHE_KEYS, ["node"]), 
        #                               RevisionsTracker())

        self._ctx = PassContext()
        self._runners = {
            PassKind.ANALYSIS: AnalysisRunner,
            PassKind.TRANSFORMATION: TransformationRunner,
        }

        self._supported_run_keywords = set()
        self._supported_pass_keywords = set()
        for i in map(_call, config.plugins):
            for plugin in i.register(self):
                if hasattr(self, name:=plugin.name):
                    raise ValueError(f'The PassManager already has an attribute named {name!r},'
                                     f' please use another name for the plugin {plugin!r}')
                setattr(self, name, plugin)
            self._supported_run_keywords.update(i.run_keywords)
            self._supported_pass_keywords.update(i.pass_keywords)
        
        # def _validate_required_tree_attributes(pp: PreparedPass):
        #     tree = pp.passe.args['tree']

        # self.hooks.install(_validate_required_tree_attributes, when='before', 
        #                    kind='transformation', level='forest')

    def configure(self, passe: PassLike, name: str) -> None:
        self._preconfigured[name] = passe._replace(name=name)
    
    def get_passe(self, name: str) -> PassLike:
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
    # is that for a given node, the passmanger has no direct knowledge of which tree
    # this node lives in. This is why in order to run a NODE analysis on anything else than the root
    # node of a tree, users of the library need to pass (at least) the tree identifer as 
    # the second argument AND the actual node instance as the third. This is why we
    # might add a simple NamedTuple class like this: 
    # class TreeNode(NamedTuple):
    #     """
    #     A class that wraps a node and under which tree it lives::
    #       tree, node = TreeNode(tree, node)
    #     """
    #     tree: Tree | RootNode | str
    #     node: AnyNode
    # This object might be returned from analyses 
    # such that is can be used like c.gather(analysis_name, *tn)


    def run(self, passe: PassLike | str, *element: str | Element, **kwargs:Any) -> CompletedPass:
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
        # :param cache_only: Forces the completed pass instance to be fetch from the cache.
        #     If the object is not found in the cache, it will raise an error.
        #     Currently only analyses end up in the cache, so make sure not to use this
        #     with a transformation. 
        """
        if len(element) > 2:
            raise TypeError("this method takes at most 3 positional arguments")
        if isinstance(passe, str):
            passe = self.get_passe(passe)
        # create the pointer
        element = self._prepare_element(element, passe.proto.runs_on)
        pointer: _ElementPath = (self.trees,) + element  # type: ignore
        runner = self._get_runner(passe(), pointer)

        # We do not validate the keywords because we might use the passmanager with unsupported
        # keywords in test cases for instance. A plugin might be written to trigger warnings
        # when issuing unsupported keywords.
        return runner.run(**kwargs)

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
        
        self.apply(add_tree(tree))

    def remove(self, tree: Tree | RootNode | str) -> None:
        """
        Remove a tree from the passmanager, 
        does nothing if the given tree is not in the system.
        """
        self.apply(remove_tree(tree))

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
                # If the first element is not a tree,
                # try to fetch it from the forest,
                # it can be either a identifier string
                # or the root node of the tree.
                # TODO: Attention: This SHOULD implicitely promotes the passe to the forest-wide
                # level when used thought the connector and the first element is not the current
                # tree . This is an intended behavior.
                module = self.trees[first_element]
                element = (module,) + element[1:]
            elif first_element not in self.trees:
                # the tree is not in the system, so add it now.
                # not that this might be run from within a passe: 
                # Which might be an analysis, in which case it's not permitted to 
                # transform the forest. A check should be added in the connector. 
                # TODO: Or maybe run raise an error ?
                self.add(first_element)
            if needs_to_append_root:
                element += (element[0].root,)

        return element

    def _get_runner(self, passe: PassInstance, pointer: _ElementPath) -> Runner:
        return self._runners[passe.proto.kind](self, passe, pointer)


# Internal builtin passes

@transformation(on=Forest)
def add_tree(_: Connector, forest: Forest, *, 
               tree: Tree) -> CastableToDict:
    """
    A forest transformation that adds the given tree.
    """
    if tree in forest:
        yield "update", False
        return
    forest._add(tree)
    yield "update", True


@transformation(on=Forest)
def remove_tree(_: Connector, forest: Forest, *, 
                  tree: Tree | RootNode | str) -> CastableToDict:
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