from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from functools import lru_cache, partial
from inspect import signature, Parameter
from itertools import chain
from typing import (TYPE_CHECKING, Callable, Collection, Hashable, Iterable, 
                    Iterator, Any, Mapping, Protocol, 
                    Sequence, TypeAlias, TypeVar, TypedDict, NotRequired, 
                    final, overload)

from libstatic._lib.structures import Cache, FrozenDict, FrozenNamespace, GetProxy, OrderedSet, CallResult


import attrs

# Represent any element of the system: forest, mtree, or any nodes
Element = Any

# Represent the root node of the tree of a module
RootNode: TypeAlias = object
# Represent any node ina module, including it's root node
AnyNode: TypeAlias = object

class MTree:
    """
    Encapsulate a single parse tree.
    
    All trees are required to have an identifier. 
    This should be the python module name.

    @note: This is a read-only datastructure. Don't try to mutate identifier, 
        root or attributes.
    """
    __slots__ = '__root', '__identifier', 'attributes'
    
    def __init__(self, root: RootNode, identifier: str, **attributes: Hashable) -> None:
        self.__root = root
        self.__identifier = identifier
        
        self.attributes: Any = FrozenNamespace(**attributes)
        """
        Optional hashable attributes metadata regarding this tree. 
        """
    
    @property
    def root(self) -> RootNode:
        return self.__root
    
    @property
    def identifier(self) -> str:
        return self.__identifier

    @lru_cache()
    def __hash__(self) -> int:
        return hash((self.root, self.identifier, self.attributes))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(self, MTree) and isinstance(other, MTree):
            return self.root == other.root and \
                self.identifier == other.identifier and \
                self.attributes == other.attributes
        return NotImplemented

class MTreeNotFound(KeyError):
    ...

class Forest(Collection[MTree]):
    """
    A collection of trees. 

    Provides a mapping interface to access the pass manager trees.

    Values can be accessed both by module name or by module ast node.

    Mutation methods (add/remove) are private since these action should only
    be performed through a transformation. You can still initiate a forest yourself
    with the constructor method, but once created it should be associated with a passmanager
    in order to further add or remove trees to/from the forest.

    >>> trees = [MTree(ast.parse(), 'mod1', filename='./mod1.py'), ...]
    >>> forest = Forest(trees)
    >>> passmanager = PassManager(forest=forest)
    """

    __slots__ = '__identifier2tree', '__root2tree', '__trees'

    def __init__(self, trees: Iterable[MTree]=None) -> None:

        # each operation must maintain these 3 structures.
        self.__identifier2tree: dict[str, MTree] = {}
        self.__root2tree: dict[RootNode, MTree] = {}
        self.__trees: OrderedSet[MTree] = OrderedSet()

        if trees is not None:
            for t in trees:
                self._add(t)
    
    def _add(self, tree: MTree) -> None:
        # no-op is the tree is already in the collection.

        if tree in self.__trees:
            return

        if tree.identifier in self.__identifier2tree:
            raise ValueError(
                f"identifier {tree.identifier!r} " 
                f"if already taken: {self[tree.identifier]}"
            )

        if tree.root in self.__root2tree:
            raise ValueError(
                f"root node {tree.identifier!r} is already "
                f"associated with another tree: {self[tree.root]}"
            )

        # add the tree in the collection.
        self.__identifier2tree[tree.identifier] = tree
        self.__root2tree[tree.root] = tree
        self.__trees.add(tree)
    
    def _remove(self, tree: MTree) -> None:
        if tree not in self:
            raise ValueError(f"tree not in the collection: {tree}")

        # remove the tree from the collection
        del self.__identifier2tree[tree.identifier]
        del self.__root2tree[tree.root]
        self.__trees.discard(tree)
    
    #  getitem interface

    def __getitem__(self, __key: str | RootNode) -> MTree:
        try:
            if isinstance(__key, str):
                return self.__identifier2tree[__key]
            else:
                return self.__root2tree[__key]
        except KeyError as e:
            raise MTreeNotFound(__key) from e
    
    def get(self, key: str | RootNode, default:Any=None) -> MTree | None:
        try:
            return self[key]
        except KeyError:
            return default
    
    # collection interface
    
    def __iter__(self) -> Iterator[MTree]:
        return iter(self.__trees)

    def __len__(self) -> int:
        return len(self.__trees)

    def __contains__(self, other: object) -> bool:
        # A forest contains trees, root nodes and identifiers.
        return other in self.__trees or \
            other in self.__identifier2tree or \
            other in self.__root2tree

PassOptions = FrozenDict
PassArgs = FrozenDict
_Json = Any

# pass kinds
class _PassKind(IntEnum): 
    TRANSFORMATION = 1; ANALYSIS = 2
# element kinds
class _ElemKind(IntEnum): 
    FOREST = 3; MTREE = 2; NODE = 1
# perf
_TRANSFORMATION = _PassKind.TRANSFORMATION
_ANALYSIS = _PassKind.ANALYSIS
_FOREST = _ElemKind.FOREST # runs on the entire Forest
_MTREE = _ElemKind.MTREE # runs on MTree instances (which includes the root node, it's identifier and metadata)
_NODE = _ElemKind.NODE  # runs on any nodes of the tree - including the root node (whiout metadata - but metadate can till be passed as pass parameters)

# # For analyses:
# def Result(value: Any) -> tuple[str, Any]:
#     return 'result', value
# def Completeness(value: bool) -> tuple[str, bool]:
#     return 'completeness', value
# # For transformations:
# def Preserved(value: PreservedAnalyses) -> tuple[str, PreservedAnalyses]:
#     return 'preserved', value
# def Update(value: bool) -> tuple[str, bool]:
#     return 'update', value

class _AnalysisReturns(TypedDict):
    result: Any
    completeness: NotRequired[bool]

class _IPassPattern(Protocol):
    """
    A pass pattern is abstracted as any object 
    which __eq__ method will return True
    for any pass that matches the pattern. 
    So a pass instance implememt this interface implicitely by default.
    """
    def __eq__(self, value: object) -> bool: ...

class _ParameterizedPassPattern:
    """
    Represents several derivations of the same pass.
    """
    
    def __init__(self, passe: PassPrototype, **args_predicate: Callable[[object], bool]) -> None:
        
        # TODO: Verify if all the parameters are given... 

        self.__match = args_predicate
        self.__passe = passe
    
    def matches(self, other: object) -> bool:
        """
        Whether the given pass instance matches the pattern.
        """
        if isinstance(other, PassInstance):
            proto = other.proto
            args = other.args
        else:
            return False
        # Two passes matches if they share the same prototype
        if proto != self.__passe:
            return False
        # And all the arguments predicates returns a truthy value,
        # for argument that are set, for the one that are eventually missing from
        # the given PassInstance, the matching is ignored.
        for k, cb in self.__match.items():
            if k in args and not cb(args[k]): 
                return False
        return True
        

PreservedAnalyses: TypeAlias = Collection['PassPrototype' | 'PassInstance' | _IPassPattern]

class _TransformationReturns(TypedDict):
    preserved: PreservedAnalyses
    update: bool # might not be necessary if we can pass a
    # value that mean preserves all results

_CastableToDict: TypeAlias = Iterable[tuple[str, Any]] | Mapping[str, Any]
"""
Anything that can be casted to  dict. 

We accept iterables so the wrapped functions can be generators::
    @analysis(on=ast.AST)
    def localsmaps(node):
        yield 'result', _fetch_locals(node)

Is equivalent to::
    @analysis(on=ast.AST)
    def localsmaps(node):
        return { 'result' : _fetch_locals(node) }

"""

class _IDoPassFunc(Protocol):
    @overload
    def __call__(self, c: IConnector, element: Element, **kwargs: Hashable) -> _CastableToDict:
        """
        A pass function might take keyword arguments, but not necessarly.
        """

# rationale around mutations of results:
# an analysis might defined mutations functions,
# the thing is all forest knowledge analysis results cannot observe
# all mutations of all modules, this would become very slow rapidly. 
# so, for now, 
# - node and tree analyses mutations won't be called if a node
#   in another tree is added or removed. 
# - all analyses mutations are called if a tree
#   in is added or removed. 
# - forest analyses mutations will be called 
#   for any added or removed node 
# - tree analyses mutations will be called 
#   for any added or removed node in the tree


class NodeOrMtreeTransMutations:
    add_node: ...
    remove_node: ...

class ForestTransMutations:
    add_mtree: ...
    remove_mtree: ...

# @attrs.frozen()
class PassPrototype:
    """
    Carries all the meta information about a pass. 

    Calling instances of this object will produce a L{PassInstance}. 
    """
    
    #: The pass function
    do_pass: _IDoPassFunc

    #: A name for this pass, this will be used in the dependencies attribute name if that's a analysis.
    name: str 
    
    #: the type of pass: transformation or analysis
    kind: _PassKind

    #: on what kind of element this pass runs on.
    runs_on: _ElemKind

    #: a isinstance check will be done and TypeError will be raised for any mismatch.
    runs_on_type: type | tuple[type, ...]

    # required parameters names declaration
    params: tuple[str, ...] = attrs.field(default=()) # at least an empty tuple
    
    # options names to their default values declaration
    optional_params: FrozenDict[str, Hashable] = attrs.field(default=FrozenDict()) # at least an empty map
    
    # a sequence of dependecies that will be bound to variable inside the 'deps' of the connector.
    dependencies: Sequence[PassOrPassProto] = attrs.field()
    
    mutations_add_node: ...
    mutations_remove_node: ...
    mutations_add_mtree: ...
    mutations_remove_mtree: ...

    # whether to cache the result of this pass in memory
    cached: bool
    
    # whether the result of this pass is always the same, 
    # like in LLVM, a pass can be marked as immutable to survive any
    # transformation implicitely because the result never depend
    # on the node it's run on, but on global constants or system information for instance.
    # TODO: implement this logic...
    immutable: bool
    
    # serialization stuff, only for analyses

    # file_cached: bool
    # # TODO: should the file caches results have a embeded version maybe?
    # encode_result: Callable[[Any], _Json]
    # decode_result: Callable[[_Json], Any]

    def __str__(self) -> str:
        # i.e. "Node analysis 'def_use_chains'" 
        return f'{self.runs_on.name.title()} {self.kind.name.lower()} {self.name!r}'

    # Methods to dynamically change a pass prototype.
    
    def set_mutations_add_node(self, f: ...): 
        # TODO: validate function.
        return self._replace(mutations_add_node=f)
    
    def set_mutations_remove_node(self, f: ...): 
        # TODO: validate function.
        return self._replace(mutations_remove_node=f)
    
    def set_mutations_add_mtree(self, f: ...): 
        # TODO: validate function.
        return self._replace(mutations_add_mtree=f)
    
    def set_mutations_remove_mtree(self, f: ...): 
        # TODO: validate function.
        return self._replace(mutations_remove_mtree=f)
    
    # private
    def _replace(self, **kwargs) -> PassPrototype:
        # TODO:...
        ...

    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        return self._instanciate()(*args, **kwargs)
    
    def proxy(self, level: _ElemKind) -> PassInstance:
        """
        Derive this pass to return new pass that results into a simple proxy that provide a C{get} method which trigers
        the original pass on the given node.
        
        This can be used to avoid calling repetitively ``passmanager.gather(pass, ...)``.
        """
        return self._instanciate().proxy(level)
    
    def _instanciate(self) -> PassInstance:
        return PassInstance(self, PassArgs())

    # Method to create a PassPattern from this pass.

    def like(self, **args_predicate: Callable[[object], bool]) -> _IPassPattern:
        """
        Create a pattern representing several possible derivations of 
        the pass to be matched against other passes. 

        Designed to be used for preserved analyses.

        When creating a "like" pattern, all parameters must be given. 

        @param kwargs: The analysis parameters names to the match function. 
            A match function is a one-argument
            callable that returne whether the value for the parameter matches.
        """
        return _ParameterizedPassPattern(
            self, **args_predicate
        )

# we need 2 decorators: 
# @analysis(on=passmanager.Forest, name='structure')
# @analysis(on=passmanager.MTree)
# @analysis(on=ast.Module)
# @transformation(on=passmanager.MTree)
# @transformation(on=ast.FunctionDef)

# forest_transformation does not exist because the only two forest
# transforms that should ever exist are adding a mtree and removing a mtree. 
_nah = object()

@attrs.frozen()
class PassInstance:
    """
    I represent a instance of a L{PassPrototype}. 

    Whereas the pass prototype, instance of this class carries the actual
    values of the parameters in use (). If the prototype do not declare any parameters,
    then this representation is redundant. 

    A pass instance have no knowledge of the node it will run onto. the pass 
    instance describe the concrete behaviour of a pass, and can be run several 
    times on several elements.
    """
    proto: PassPrototype
    args: PassArgs
    
    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        newpass = self
        if args or kwargs:
            newpass = newpass._add_args(*args, **kwargs)
        return newpass

    def _add_args(self, *args: Hashable, **kwargs: Hashable) -> PassInstance: 
        if len(kwargs) > len(optional_params:=self.proto.optional_params):
            raise TypeError(f'too many keyword parmeters, expected at most {len(optional_params)} keywords')
        if len(args) > len(params:=self.proto.params):
            raise TypeError(f'too many positional parmeters, expected at most {len(params)} positionals')
        
        self_args = self.args
        args_dict = {}
        # support positional parameters
        for pname, value in zip(params, args):
            # This prevents the creation of new instance of PassInstance whith the same params values.
            if value != (self_args.get(pname, _nah)):
                args_dict[pname] = value
        
        for pname, value in tuple(kwargs.items()):
            if pname not in params or pname not in optional_params:
                raise TypeError(f'unexpected argument {pname}')
            if pname in args_dict:
                raise TypeError(f'got several values for parameter {pname}')
            if value != self_args.get(pname, _nah):
                args_dict[pname] = value
        
        if args_dict:
            return PassInstance(
                self.proto, 
                FrozenDict({**self.args, **args_dict})
            )
        else:
            return self

    def proxy(self, level: _ElemKind) -> PassInstance:
        if self.proto.kind != _ANALYSIS:
            # Client need to write their own wrapper for transformation
            # to run it on all applicable nodes of a module for instance. This is 
            # a task that cannot be generalized for all tree types so it doesn't belong here.
            raise TypeError('cannot proxy a transformation')
        if runs_on:=self.proto.runs_on == _FOREST:
            raise ValueError('cannot proxy a forest analysis')
        if level < runs_on:
            raise ValueError('cannot proxy an analysis to a lower level')
        if level == runs_on:
            # that's a unssuported special case because... 
            raise ValueError('cannot proxy an analysis to the same level')
        if level == _FOREST:
            # To create the proxy we need to dynamically change it's prototype
            # in order to have the same name as the proxied analysis.
            new_proto = _forest_proxy_pass._replace(name=self.proto.name)
        elif level == _MTREE:
            new_proto = _mtree_proxy_pass._replace(name=self.proto.name)
        else:
            assert False
        return new_proto(__proxied__=self)

    def missing_param(self) -> str | None: 
        if(any((missing:=p) not in self.args for p in self.proto.params)):
            return missing
        return None
    
    # def proxied(self) -> bool:
    #     return '__proxied__' in self.proto.params
    
    # def cached(self) -> bool: 
    #     return self.proto.cached

    def get_dependencies(self) -> Collection[PassOrPassProto]:
        return self.proto.dependencies

    def get_all_dependencies(self) -> Collection[PassOrPassProto]:
        seen = OrderedSet()
        def _yield_deps(c: PassOrPassProto):
            yield from (d for d in c.get_dependencies() if d not in seen)
            yield from (d for d in chain.from_iterable(
                _yield_deps(dep) for dep in c.get_dependencies()) if d not in seen)
        seen.update(_yield_deps(self))
        return seen
    
    def _replace(self, **kwargs) -> PassInstance:
        ...

PassOrPassProto: TypeAlias = PassPrototype | PassInstance

_posargs = frozenset((Parameter.POSITIONAL_OR_KEYWORD, 
                          Parameter.POSITIONAL_ONLY, ))

_runs_on_type_2_level = {Forest: _FOREST, MTree: _MTREE}

def new_pass_prototype(
        do_pass: Callable[..., _CastableToDict], 
        *, 
        kind: _PassKind, # analysis or transformation
        on: type | Iterable[type], 
        dependencies: Sequence[PassOrPassProto] | None = None,
        cached: bool = True,
        immutable: bool = False,
        # file_cached: bool = False,
    ) -> PassPrototype: 
    """
    
    @param do_pass: A callable that contains the driving logic of your pass.
      By convention, the callable should be a generator function yielding tuples: (key, value). 
      But the
      
      A passe can provide metadata that are not directly
      meant to be presented to the users but rather use to internally optimize runs.
    
      Support four variants for function based: 
        
        - with connector, two posargs::
            def f(c, node): ...
        - with connector, two posargs and x keywords::
            def f(c, node, *, arg1, arg2=False): ...
        - without connector, one posarg::
            def f(node): ...
        - without connector one posarg and x keywords::
            def f(node, *, arg1, arg2=False): ...
    
    Support four variants for class based, used for better typing experience: 

        @transformation(on=ast.Module)
        class my_pass(Transforms[ast.Module]):
            def run(c, node: ast.Module)
        
        
    
    @param kind: L{ANALYSIS} or L{TRANSFORMATION}.
    @param on: The type of the element the pass is supposed to be run on. 
        I.e. L{Forest}, L{MTree}, L{ast.Module}, L{ast.FunctionDef}. 
        This can also be a tuple of types, but this is only applicable 
        if your pass runs on syntax tree nodes (not MTree or Forest). 
    @param dependencies:
    @param cached:
    @param immutable:
    """
    try:
        do_pass_sig = signature(do_pass)
    except Exception as e:
        raise TypeError('only pure-python functions are supported at the moment') from e
    
    # Validate the signature...

    # Dynamically build the parameters
    params: list[str] = []
    optional_params: dict[str, Hashable] = {}
    
    nb_pos_args = 0
    for param_name, param in do_pass_sig.parameters.items():
        if param.kind in _posargs: 
            nb_pos_args += 1
            if nb_pos_args > 2:
                raise TypeError(
                    'A pass function must not take more '
                    'than two positional arguments please use keyword-only arguments.')
        elif (default:=param.default) is not Parameter.empty:
            params.append(param_name)
        else:
            optional_params[param_name] = default
    if nb_pos_args == 0:
        raise TypeError(
                    'A pass function must take at least  '
                    'one positional argument.')
    elif nb_pos_args == 1:
        # wrap functions that do not take a connector inside one that ignores it. 
        actual_do_pass = lambda _, node, **kws: do_pass(node, **kws)
    else:
        assert nb_pos_args == 2
        actual_do_pass = do_pass

    # Determine the runs_on_type based on provided 'on' param.
    runs_on_type = on
    if not isinstance(runs_on_type, type):
        if not isinstance(runs_on_type, tuple):
            runs_on_type = tuple(runs_on_type)
        if len(runs_on_type)==0:
            raise ValueError('parameter "on" cannot be empty')
        if len(runs_on_type)!=1:
            # Validate the value since MTree and Forest should not be present in 
            # passes that run on several nodes. This is a limitation that is necessary by design
            # since it is used used to differenciate a FOREST pass and a MTREE pass etc.
            if any((problematic:=t) in _runs_on_type_2_level for t in runs_on_type):
                raise TypeError(f'a pass cannot run both on {problematic} and on other types')
        else:
            # only one value, so flatten it
            runs_on_type, = runs_on_type

    # Determine the runs_on_level
    runs_on_level = _runs_on_type_2_level.get(runs_on_type, _NODE)

    return PassPrototype(
        actual_do_pass, 
        name=do_pass.__name__, 
        kind=kind,
        runs_on=runs_on_level, 
        runs_on_type=runs_on_type,
        params=params, 
        optional_params=optional_params,
        dependencies=dependencies, 
        cached=cached,
        immutable=immutable,
    )
    

def _pass_decorator(**kwargs):
    def decorator(function):
        return new_pass_prototype(function, **kwargs)
    return decorator

# main decorators to create a transformation
transformation = partial(_pass_decorator, kind=_TRANSFORMATION)
# main decorators to create an analysis
analysis = partial(_pass_decorator, kind=_ANALYSIS)

@analysis(on=Forest, cache=False)
def _forest_proxy_pass(c: IConnector, _: Forest, *, 
                       __proxied__: PassInstance) -> _CastableToDict:
    def inner_pass(*element, **kwargs):
        if kwargs:
            runpass = __proxied__(**kwargs)
        else:
            runpass = __proxied__
        return c.gather(runpass, *element)
    return {'result':GetProxy(inner_pass)}

@analysis(on=MTree, cache=False)
def _mtree_proxy_pass(c: IConnector, node: MTree, *, 
                      __proxied__: PassInstance) -> Iterable[str, GetProxy]:
    def inner_pass(element, **kwargs):
        if kwargs:
            runpass = __proxied__(**kwargs)
        else:
            runpass = __proxied__
        return c.gather(runpass, node, element)
    return {'result':GetProxy(inner_pass)}


_RunsOnPointer: TypeAlias = tuple[Forest,] | tuple[Forest, MTree] | tuple[Forest, MTree, AnyNode]
"""
A "pointer" stores the path of an element under one of these forms: 
    
    - forest
    - forest, mtree
    - forest, mtree, node
"""

_PassRun: TypeAlias = tuple[PassInstance, _RunsOnPointer]
"""
A "pass run" stores a pass and on which element it has been run.
"""

class _PassRunMetadata:
    __slots__ = 'knowledge',
    knowledge: _ElemKind | None = None

class PassContext:
    """
    Class that does the book-keeping of the chains of passes runs.
    """

    __slots__ = '_knowledge_stack',

    # maintains a "stack" of running passes and on which element
    # the stack is implemented as dict because it stores the knowledge
    # level of the pass run as well.
    def __init__(self) -> None:
        self._knowledge_stack: dict[_PassRun, _ElemKind] = {}
    
    @property
    def _current_passrun(self) -> _PassRun:
        try:
            return next(reversed(self._knowledge_stack))
        except StopIteration:
            raise RuntimeError('no pass is currently running')

    @property
    def current(self) -> Forest | MTree | AnyNode:
        return self._current_passrun[1][-1]

    @contextmanager
    def _push_pass(self, passe: PassInstance, pointer: _RunsOnPointer) -> Iterator[_PassRunMetadata]:
        
        key: _PassRun = (passe, pointer)
        
        if key in self._knowledge_stack:
            # TODO: Use a exception subclass in order to potentially catch and
            # ase another strategy for analysing this node.
            raise RuntimeError(f'cycle detected with pass: {key}')

        self._knowledge_stack[key] = passe.proto.runs_on
       
        # Might be interesting to optimize the remove mtree transformation: 
        # Yield a context tracker that is able to say which knowledge 
        # the pass accessed as well as the complete list of dependent mtrees 
        # in the case of a forest knowledge analysis.
        # We can do this safely only if no direct access to the forest is done.
        # Forest proxies can be used to gather info for a different module but 
        # the pass should never directly read the content of the Forest. 
        meta = _PassRunMetadata()
        yield meta
        
        # this is just in case someone does something stupid
        if __debug__:
            e = next(reversed(self._knowledge_stack))
            if e is not key:
                raise RuntimeError(f'pass context is confused: {e} is not {key}')
            del e

        # pop element from "stack"
        meta.knowledge = pass_run_knowledge = self._knowledge_stack[key]
        del self._knowledge_stack[key]

        # so at this time pass_run_knowledge contains the maximum runs_on level of the "passe"
        # and all it's used dependencies.
        
        # propagate the knowledge of the dependency pass towards the calling pass if any.
        if self._knowledge_stack:
            curr = self.current
            if self._knowledge_stack[curr] < pass_run_knowledge:
                self._knowledge_stack[curr] = pass_run_knowledge

class IConnector(Protocol):
    """
    Connector to the passmanager, from inside a pass function.
    This is what we get as the first argument of pass functions like::

        @analysis(on=AST)
        def stuff(connector: IConnector, node): ...
    """
    deps: Dependencies
    gather: Callable[..., Any] #: See L{PassManager.gather}

    if "it's connected to a transformation":
        apply: Callable[..., Any] #: See L{PassManager.apply}

        # TODO: implement me
        # mutations: NodeOrMtreeTransMutations | ForestTransMutations


CACHE_KEYS: frozenset[str] = OrderedSet(
    (
     'passe', # PassInstance

     'knowledge', # integer: FOREST / MTREE / NODE
     'completeness',  # boolean
     
     'mtree', # MTree or None
     'node', # AnyNode or None
     )
)

_CacheKeyT: TypeAlias = tuple[
    PassInstance, 

    int, 
    bool, 
    
    Hashable | None, 
    Hashable | None
    ]


def _mk_cache_key_to_set_result(passe: PassInstance, 
                              pointer: _RunsOnPointer, 
                              analysis_result: CallResult[_AnalysisReturns], 
                              knowledge_level: _ElemKind) -> _CacheKeyT:
    
    # we do not use the forest part of the pointer here
    path = pointer + (None, ) * (3 - len(pointer))
        
    if knowledge_level == _FOREST:
        completeness = analysis_result.result.get('completeness', False)
    else:
        completeness = True
    
    return passe, int(knowledge_level), completeness, path[1], path[2]
    

def _mk_cache_keys_to_get_result(passe:PassInstance, 
                                  pointer: _RunsOnPointer, ) -> Iterator[_CacheKeyT]:     
    path = pointer + (None, ) * (3 - len(pointer))
    p1 = path[1]
    p2 = path[2]
    runs_on = int(passe.proto.runs_on)
    
    yield passe, runs_on, True, p1, p2,

    while runs_on < _ElemKind.FOREST:
        runs_on += 1
        yield passe, runs_on, True, p1, p2,
    
    # The completeness can only be False for forest knowledge analyses.
    yield passe, runs_on, False, p1, p2,

class Runner:

    __slots__ = '_passmanager', '_passe', '_pointer', 
    
    @final
    def __init__(self, 
                 passmanager: PassManager, 
                 passe: PassInstance, 
                 pointer: _RunsOnPointer) -> None:
        self._passmanager = passmanager
        self._passe = passe
        self._pointer = pointer

    @contextmanager
    def push(self) -> Iterator[_PassRunMetadata]:
        with self._passmanager._ctx._push_pass(self._passe, self._pointer) as meta:
            yield meta

    @final
    def prepare(self) -> IConnector:
        
        # prepare the pass connector namespace
        p = self._passe
        passe_proto = p.proto
        element = self._pointer[1:]

        # validate the runtime type
        if runs_on_type:=passe_proto.runs_on_type:
            if not isinstance(element, runs_on_type):
                raise TypeError(f'unexpected type, got {type(element)}, should be {runs_on_type}')

        # Apply all transformations eagerly, since we use a descriptor for all analyses
        # we need to transitively iterate dependent transforms and apply then now.
        for _t in p.get_all_dependencies():
            t_proto = _t if isinstance(_t, PassPrototype) else _t.proto
            
            if t_proto.kind != _PassKind.TRANSFORMATION:
                continue
            if (t_runs_on:=t_proto.runs_on) < (p_runs_on:=passe_proto.runs_on):
                # Since a NODE pass can be run on a MTree, we should explicitely
                # accept if we hit this case. BUT what if the transformation only 
                # applies to FunctionDef for instance ? well.. then it will fail
                # at the apply() stage with a TypeError. 
                # TODO: we might be able to check the config to see if pass can be run
                # on root nodes or not.
                if t_runs_on == _NODE and p_runs_on == _MTREE:
                    apply_on_element = element

                else:
                    # it's not clear from the code but that's the only 
                    # possible value for this variable at this point. 
                    if __debug__: assert p_runs_on == _FOREST

                    # TODO: This is a rather important limitation, we might be able to overcome
                    # it by providing a wrapper function to the transformation prototype. 
                    # and a wrapper() method that will work a bit like proxy()
                    # something like
                    # @transformation(on=ast.FunctionDef)
                    # def add_explicit_returns_None(c, node): ...
                    # @add_explicit_returns_None.set_wrapper(on=Forest)
                    # def _(c, forest):
                    #   for all functions of forest: c.apply(add_explicit_returns_None, block)
                    # @add_explicit_returns_None.set_wrapper(on=Tree)
                    # def _(c, tree):
                    #   for all functions in tree: c.apply(add_explicit_returns_None, block)
                    # this will fail: 
                    # pm.apply(add_explicit_returns_None, 'testmodule')
                    # this will be successful
                    # pm.apply(add_explicit_returns_None, 'testmodule', some_func_node)
                    # pm.apply(add_explicit_returns_None.wrapper(Tree), 'testmodule')
                    # pm.apply(add_explicit_returns_None.wrapper(Forest))
                    # END TODO:

                    raise TypeError(f'{p} cannot depend - even transitively - on {_t}. '
                                    'A pass can only depend on transformations '
                                    'that runs on a compatible or enclosing level, '
                                    'try using wrapper(). ')
            else:

                lvldiff = t_runs_on - p_runs_on
                apply_on_element = element[:-lvldiff]
            
            self._passmanager.run(_t, *apply_on_element)                    

        # create the analysis dependencies namespace.
        deps = Dependencies()
        for _a in p.get_dependencies():
            a_proto = _a if isinstance(_a, PassPrototype) else _a.proto
            if a_proto.kind != _PassKind.ANALYSIS:
                continue
            if (a_runs_on:=a_proto.runs_on) < (p_runs_on:=passe_proto.runs_on):
                if a_runs_on == _NODE and p_runs_on == _MTREE:
                    dep_element = element

                else:
                    # this is true because we have only 3 levels of elements.
                    if __debug__: assert a_runs_on == _FOREST 
                    
                    raise TypeError(f'{p} cannot depend on {_a}, '
                                    'a pass can only depend on analyses '
                                    'that runs on a compatible or enclosing level, '
                                    'try using proxy().')
            else:
                lvldiff = a_runs_on - p_runs_on
                dep_element = element[:-lvldiff]
            
            if _missing:=_a.missing_param():
                # dependency is missing a required parameter and cannot be presented as a descriptor. 
                # Instead of trying to do something smart and complex, we fail early and propose the user
                # to use proxy() which supports passing the pass arguments as .get() keywords.

                raise TypeError(f'{p} cannot depend on {_a} because it is missing '
                                f'a required parameter {_missing!r} '
                                'try using proxy() and pass that parameter value as keyword.')
               
                # So for instance a NODE analysis 'attribute' which require a 'name' parameter 
                # can be included in the dependency list like that::
                # @analysis(deps=[attribute.proxy(Forest/MTree)])
                # def my_pass(c, node):
                #   c.deps.attribute('some_module_name', name='some_name')
                #   c.deps.attribute('some_module_name', class_def, name='some_name')
                # Which is a little bit nicer than using gather() direclty, but pass args
                # needs to be supplied by keyword.
                # @analysis()
                # def my_pass(c, node):
                #   c.gather(attribute('some_name'), 'some_other_module_name')
                #   c.gather(attribute('some_name'), 'some_other_module_name', class_def)
            else:
                # the dependency can be converted to a descriptor
                # TODO: I'm sure there is a faster way to do it
                callback: Callable[[], Any] = partial(self._passmanager.run, _a, *dep_element)
                setattr(deps, a_proto.name, _PassDependencyDescriptor(callback))
        
        # create the namespace
        if passe_proto.kind == _ANALYSIS:
            connector = FrozenNamespace(
                deps = deps, 
                gather = self._passmanager.gather
            )
        else:
            connector = FrozenNamespace(
                deps = deps, 
                gather = self._passmanager.gather, 
                apply = self._passmanager.apply,
                # TODO: implement...
                mutations = ForestTransMutations() if \
                    passe_proto.runs_on==_FOREST else NodeOrMtreeTransMutations(),
            )

        return connector

    @final
    def do_pass(self, c: IConnector) -> Any:
        # call the pass function
        p = self._passe
        return p.proto.do_pass(c, self._pointer[-1], **p.args)

    def run(self):
        raise NotImplementedError()

class AnalysisRunner(Runner):
    
    def run(self):
        p = self._passe
        pointer = self._pointer
        analysis_actually_ran = False

        try:
            with self.push() as meta:
                analysis_result = None
                if p.proto.cached:
                    for k in _mk_cache_keys_to_get_result(p, pointer):
                        analysis_result: CallResult = self._passmanager.cache.get(k)
                        if analysis_result is not None:
                            break
                if analysis_result is not None:  # the result is cached
                    # will raise an error if the initial analysis raised
                    return analysis_result.result['result'] 

                try:
                    r = self.do_pass(self.prepare())
                    if not isinstance(r, Mapping):
                        r = dict(r)
                    try:
                        result = r['result']
                    except KeyError as e: 
                        raise TypeError(f'{p} did not yield any result') from e
                    analysis_result: CallResult[_AnalysisReturns] = CallResult.new(r)
                    analysis_actually_ran = True
                    return result
                
                except Exception as e:
                    analysis_result = CallResult.new(e)
                    raise
 
        finally:
            # Set the analysis result in the cache once we have left the with: block.
            if analysis_actually_ran and p.proto.cached:
                level = meta.knowledge
                assert level is not None
                k = _mk_cache_key_to_set_result(p, self._pointer, 
                                                analysis_result, level)
                self._passmanager.cache.set(k, analysis_result)

class TransformationRunner(Runner):
    
    def run(self):
        runs_on = self._passe.proto.runs_on
        
        with self.push():

            tr =  self.do_pass(self.prepare())
            if not isinstance(tr, Mapping):
                tr = dict(tr)
            
            # stuff that needs to be invalidated, depending on the element type we're changing
            if runs_on == _FOREST:
                # this block is very special because FOREST transformations can only two kind of things:
                # an addition or a removal of MTree. 
                # - For a forest transformation: 
                #   - module added: 
                #       - All forest knowledge analyses that are not complete
                #   - module removed, not really optimized:
                #       - All forest knowledge analyses
                
                # We can't possiblily know in advance which mtree 
                # a certain forest knowledge analysis will depend on when it will be ran.
                # the import graph does not carry all the information necessary to be certain 
                # an analysis might request modules that are not in the dependant of the current module
                ...
                
            elif runs_on == _MTREE:
                # - For a mtree transformation: 
                # 
                # - all forest knowledge analyses except all that do not depend on this MTree as well as few preserved
                ...
            elif runs_on == _NODE:
                ...
            else:
                assert False

class PassManagerConfig(Protocol):
    root_node_type: type[Any]
    tree_attributes: list[str]
    child_nodes: Callable[[AnyNode], Iterable[AnyNode]]

class ASTConfig(PassManagerConfig):
    import ast
    root_node_type = ast.Module
    tree_attributes = ['filename', 'is_package', 'is_stub', 'code']
    child_nodes = ast.iter_child_nodes

class PassManager: 

    def __init__(self, 
                 forest: Forest | None = None, 
                 *, 
                 config: PassManagerConfig = ASTConfig) -> None:

        self.forest = forest or Forest()
        self.config = config
        
        self.cache: Cache[_CacheKeyT, CallResult] = Cache(CACHE_KEYS, ['node'])
        
        self._ctx = PassContext()
        self._runners = {_ANALYSIS: AnalysisRunner, 
                         _TRANSFORMATION: TransformationRunner}

    def apply(self, transform: PassOrPassProto, *element: Element) -> Any:
        proto = transform if isinstance(transform, PassPrototype) else transform.proto
        if proto.kind != _TRANSFORMATION:
            raise TypeError
        return self.run(transform, *element)

    def gather(self, analysis: PassOrPassProto, *element: Element,) -> Any:
        proto = analysis if isinstance(analysis, PassPrototype) else analysis.proto
        if proto.kind != _ANALYSIS:
            raise TypeError
        return self.run(analysis, *element)
   
    def _prepare_element(self, element: tuple[Element,...], runs_on: _ElemKind) -> tuple[Element,...]:
        if element and not isinstance(element[0], MTree):
            # If the first element is not a mtree, 
            # try to fetch it from the forest.
            module = self.forest[element[0]]
            element = (module, ) + element[1:]
        len_element = len(element)
        if runs_on == _NODE:
            if len_element == 1:
                # Very important for usability!!!
                # a NODE pass can be run on a MTree, 
                # in this case use the root module as the node.
                element += (element[0].root, )

            elif len_element == 0:
                raise TypeError('a NODE pass expect at least one element argument.')
        elif runs_on == _MTREE:
            if len_element == 0:
                raise TypeError('a MTREE pass expect one element argument.')
            elif len_element == 2:
                raise TypeError('a MTREE pass expect exactly one element argument, got 2.')
        elif runs_on == _FOREST:
            if len_element != 0:
                raise TypeError(f'a FOREST pass do not expect any element argument, got {len_element}.')
        
        return element

    def _get_runner(self, passe: PassInstance, pointer: _RunsOnPointer) -> Runner:
        return self._runners[passe.proto.kind](self, passe, pointer)

    def run(self, passe: PassOrPassProto, *element: Element) -> Any:
        """
        
        :param passe: A Pass instance or a pass prototype.
        :param element: The element on which to run the pass.
            - Zero element arguments will run the pass on the entire forest.
            - First element argument should be the module element, if only one element is given 
              it will run the pass on the specified module
            - Second element argument should be the node instance, if both module and node elements are given, 
              it will run the pass on the specified node which should live inside the specified module.
        """
        if len(element) > 2:
            raise TypeError('this method takes at most 3 positional arguments')
        
        # this method is ment to be higher level so we accept prototypes as well for easy of use
        # so we need to instanciate them manually now if needed.
        if isinstance(passe, PassPrototype):
            p = passe._instanciate() 
        elif isinstance(passe, PassInstance):
            p = passe
        else: 
            raise TypeError(f'unexpected type {type(passe)}')

        # create the pointer
        element = self._prepare_element(element, p.proto.runs_on)
        pointer: _RunsOnPointer = (self.forest, ) + element

        runner = self._get_runner(p, pointer)
        return runner.run()

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
    def __getattribute__(self, name):
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            return attr.callback()
        return attr

# class PassRunner:
#     ...

    # run()
    # prepare()
    # gather()
    # apply()
    # etc...

# pm = PassManager()
# assert isinstance(pm.forest, Forest)
# pm.add(MTree('builtins', root=ast.parse(...)))
# pm.add(MTree('typing', root=ast.parse(...)))
# module = pm.forest['builtins']
# for n in (n for n in ast.walk(module.root)):
#    local_vars = pm.gather(local_variables, 'builtins', n)

# A analysis dependency is only a wrapper for calling gather
# with partial arguments already added.
# So the potential dependencies matrix is something like

### Generic

# NODE pass has NODE dep
# NODE pass has NODE dep as proxy(MTREE)
# NODE pass has NODE dep as proxy(FOREST)
# NODE pass has MTREE dep
# NODE pass has MTREE dep as proxy(FOREST)
# NODE pass has FOREST dep

# MTREE pass has NODE dep as proxy(MTREE) only
# MTREE pass has MTREE dep
# MTREE pass has MTREE dep as proxy(FOREST)
# MTREE pass has FOREST dep

# FOREST pass has NODE dep as proxy(FOREST) only
# FOREST pass has MTREE dep as proxy(FOREST) only
# FOREST pass has FOREST dep
