from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from inspect import signature, Parameter
from itertools import chain
import itertools
from typing import (Callable, Collection, Container, Hashable, Iterable, 
                    Iterator, Any, Literal, Mapping, Protocol, Sequence, 
                    TYPE_CHECKING, Tuple, TypeVar, overload)
if TYPE_CHECKING:
    from typing import NotRequired, TypeAlias, TypedDict, final
else:
    final = lambda f: f
    TypedDict = object

from libstatic._lib.structures import (Cache, FrozenDict, FrozenNamespace, 
                                       GetProxy, OrderedSet, ChainSet)

import attrs

# TODOs: 
# - Rename MTree to simply Tree
# - The Pointer type seems redundant since we actually 
# don't need the forest to be passed arround
# because we ever operate on a single forest.
# Remove support for proxy(), it's not worth it...

############ Typing related declarations

Element: TypeAlias = Any; 'Represent any element of the system: forest, mtree, or any nodes'
RootNode: TypeAlias = object; 'Represent the root node of the module (typically ast.Module)'
AnyNode: TypeAlias = object; "Represent any node in a module, including it's root node"

PassLike: TypeAlias = 'PassPrototype | PassInstance'
CastableToDict: TypeAlias = 'Iterable[tuple[str, Any]] | dict[str, Any]'
"""
Anything that can be casted to dict. 

We accept iterables so the wrapped functions can be generators
yielding tuples: key, value ::
    @analysis(on=ast.AST)
    def localsmaps(node):
        yield 'result', _fetch_locals(node)

Is equivalent to::
    @analysis(on=ast.AST)
    def localsmaps(node):
        return dict(result=_fetch_locals(node))

"""

class IPassFunction(Protocol):
    """
    A pass function is a two positional argument 
    function with optionnaly any keyword arguments.
    """
    def __call__(self, c: Connector, element: Element, **kwargs: Hashable) -> CastableToDict:
        ...

class AnalysisReturn(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
      returned from a B{analysis} function.

    @note: If you wish better type checking of your function bodies, 
        only returns actual dict instances and use this class as 
        return type of the function. I haven't figured out yet how to 
        properly type-check this kind of code...
    """
    result: Any
    completeness: NotRequired[bool]

class TransformationReturn(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs) 
    returned from a B{transformation} function.
    """
    update: bool
    preserved: NotRequired[list[PassLike | PassPattern]]

_Pointer: TypeAlias = 'tuple[Forest,] | tuple[Forest, MTree] | tuple[Forest, MTree, AnyNode]'
"""
A "pointer" tuple stores the path of an element in the system under one of these forms: 
    
    - forest
    - forest, mtree
    - forest, mtree, node
"""
_SimplePointer: TypeAlias = 'tuple[()] | tuple[MTree,] | tuple[MTree, AnyNode]'
"""
The "simple pointer" is what's left from the "pointer" when we remove the forest.
"""

_PassRun: TypeAlias = 'tuple[PassInstance, _Pointer]'
"""
A "pass run" stores a pass and on which element it has been run.
"""

_CacheKeyT: TypeAlias = Tuple[
    'PassInstance', 

    int, 
    bool, 
    
    'Hashable | None', 
    'Hashable | None'
    ]

class Hook(Protocol):
    """
    A hook is callable that is used to customize the logic just after a pass has been run. 
    Note that the hooks won't be called when retreiving results from the cache; only when a pass actually runs.

    If the hook function returns a truthy value, it'a assumed to replace the given returned dictionary. 
    You can also mutate the dict in-place and return a falsy value.
    """
    def __call__(self, passe: PassInstance, pointer: _Pointer, returned: TransformationReturn | AnalysisReturn, ) -> TransformationReturn | AnalysisReturn | None:
        ...

_T = TypeVar('_T')

############ Actual framework

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
        
        # TODO: This should be a mapping instead.
        self.attributes: Any = FrozenNamespace(**attributes)
        """
        Optional hashable metadata regarding this tree. 
        """
    
    @property
    def root(self) -> RootNode:
        """
        The root node of the tree.
        
        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = MTree(astmod, 'builtins')
        >>> assert tree.root is astmod
        """
        return self.__root
    
    @property
    def identifier(self) -> str:
        """
        The identifier of the module. 

        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = MTree(astmod, 'builtins')
        >>> assert tree.identifier == 'builtins'
        >>> assert tree == MTree(astmod, 'builtins')
        >>> assert tree != MTree(ast.parse('...'), 'builtins')
        """
        return self.__identifier

    def __hash__(self) -> int:
        return hash((self.root, self.identifier, self.attributes))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(self, MTree) and isinstance(other, MTree):
            return self.root == other.root and \
                self.identifier == other.identifier and \
                self.attributes == other.attributes
        return NotImplemented

class MTreeNotFound(KeyError):
    """
    A subclass of KeyError that is used when a tree is not found in the forest. 
    """

class Forest(Collection[MTree]):
    """
    A collection of trees. 

    Provides a mapping-ish interface to access the pass manager trees. 
    You should not initiate a forest yourself, the passmanager will do it. 

    Values can be accessed both by module name or by module ast node.

    Mutation methods (add/remove) are private since these action should only
    be performed through the passmanager. 

    >>> import ast
    >>> astmod = ast.parse('x = 1')
    >>> tree = MTree(astmod, 'mod1', filename='./mod1.py')
    >>> pm = PassManager([tree])
    >>> isinstance(pm.trees, Forest)
    True
    >>> 'mod1' in pm.trees
    True
    >>> astmod in pm.trees
    True
    >>> tree in pm.trees
    True

    """

    __slots__ = '__identifier2tree', '__root2tree', '__trees'

    def __init__(self, trees: Iterable[MTree] | None = None) -> None:

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
    
    @overload
    def get(self, key: str | RootNode) -> MTree | None: ...
    @overload
    def get(self, key: str | RootNode, default:_T) -> MTree | _T: ...
    def get(self, key: str | RootNode, default:Any=None) -> Any:
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

# pass kinds
class _PassKind(IntEnum): "transformation or analysis"; TRANSFORMATION = 1; ANALYSIS = 2
# element kinds
class _ElemKind(IntEnum): "forest, tree or node"; FOREST = 3; MTREE = 2; NODE = 1
# perf
_TRANSFORMATION = _PassKind.TRANSFORMATION
_ANALYSIS = _PassKind.ANALYSIS
_FOREST = _ElemKind.FOREST # runs on the entire Forest
_MTREE = _ElemKind.MTREE # runs on MTree instances (which includes the root node, it's identifier and metadata)
_NODE = _ElemKind.NODE  # runs on any nodes of the tree - including the root node (whithout metadata - but metadate can till be passed as pass parameters)

class PassPattern:
    """
    Represents several derivations of the same pass prototype, 
    with different parameters. 
    Use the L{like()} method to create instances of this class. 

    >>> import ast
    >>> @analysis(on=ast.AST)
    ... def count_objs(_, node: ast.AST, *, type:type=object):
    ...     yield 'result', len(set(n for n in ast.walk(node) if isinstance(n, type)))
    
    Let's declare a few testing patterns: 

    >>> like_none = count_objs.like(type=(lambda _: False)) # this pattern will never match any pass instance
    >>> like_any = count_objs.like(type=(lambda _: True))   # this pattern will match any instances of the pass 'count_objs'
    >>> like_any2 = count_objs.like()                       # this one idem
    >>> like_cls = count_objs.like(type=(lambda v: v is ast.ClassDef)) # this one will match only if type=ast.ClassDef. 

    The pass pattern implement C{__eq__} wich makes it suitable for 
    checking whether a pass instance is C{in} a container with a pattern in it.

    >>> count_objs() in [like_cls]
    False
    >>> count_objs(type=ast.AST) in [like_none]
    False
    >>> count_objs(type=ast.AST) in [like_any]
    True
    >>> count_objs(type=ast.AST) in [like_any2]
    True
    >>> count_objs(type=ast.AST) in [like_cls]
    False
    >>> count_objs(type=ast.ClassDef) in [like_cls]
    True

    A pass prototype never macthes against a pattern, 
    only actual pass instances matches. 
    
    >>> count_objs in [like_none, like_any2, like_any, like_cls]
    False

    The matching is ignored for missing arguments. 

    >>> @analysis(on=ast.AST)
    ... def has_required_param(_, node, *, required):
    ...     yield 'result', 1
    >>> param_like_none = has_required_param.like(required=(lambda _: False))
    >>> has_required_param() in [param_like_none]
    True
    >>> has_required_param('anything') in [param_like_none]
    False

    The pattern can be generic by ommiting the argument.

    >>> has_required_param('anything') in [has_required_param.like()]
    True
    """

    # This class is not hashable by nature.

    __slots__ = '_match', '_passe'
    
    def __init__(self, passe: PassPrototype, **predicate: Callable[[object], bool]) -> None:
        # TODO: Not all parameters might be given, 
        # but we should still validate the name of the predicates!!!
        self._match = predicate
        self._passe = passe
    
    def matches(self, other: object) -> bool:
        """
        Whether the given pass instance matches the pattern.
        """
        if not isinstance(other, PassInstance):
            return False
        proto = other.proto
        # Two passes matches if they share the same prototype
        if proto != self._passe:
            return False
        
        args = {**proto.optional_params, **other.args}
        # And all the arguments predicates returns a truthy value,
        # for argument that are set, for the one that are eventually missing from
        # the given PassInstance, the matching is ignored.
        for k, cb in self._match.items():
            if (k in args) and (not cb(args[k])): 
                return False
        return True
    
    __hash__ = None
    __eq__ = matches
        
@attrs.frozen()
class PassPrototype:
    """
    Carries all the meta information about a pass. 

    Calling instances of this object will produce a L{PassInstance}. 
    """
    
    #: The pass function
    do_pass: IPassFunction

    #: A name for this pass, this will be used in the 
    #: dependencies attribute name if that's a analysis.
    name: str 
    
    #: the type of pass: transformation or analysis.
    kind: Literal[_PassKind.ANALYSIS, _PassKind.TRANSFORMATION]

    #: on what kind of element this pass runs on? this is conceptual.
    runs_on: Literal[_ElemKind.NODE, _ElemKind.MTREE, _ElemKind.FOREST]

    #: on what type of object this pass runs on? 
    #: a isinstance check will be done and L{TypeError} 
    #: will be raised for any mismatch.
    runs_on_type: type | tuple[type, ...]

    # required parameters names declaration
    params: tuple[str, ...] = attrs.field(default=(), converter=tuple) # at least an empty tuple
    
    # options names to their default values declaration
    optional_params: Mapping[str, Hashable] = attrs.field(default=FrozenDict(), converter=FrozenDict) # at least an empty map
    
    # a sequence of dependecies that will be bound to variable inside the 'deps' of the connector.
    dependencies: Sequence[PassLike] = attrs.field(default=(), converter=tuple) # at least an empty tuple
    
    #: whether to cache the result of this pass in memory
    cached: bool = True
    
    # whether the result of this pass is always the same, 
    # like in LLVM, a pass can be marked as immutable to survive any
    # transformation implicitely because the result never depend
    # on the node it's run on, but on global constants or system information for instance.
    # TODO: implement this logic...
    # TODO: Maybe this could be renamed to longlived, so that it's clear that it can be used for real
    # anbalyses that auto-updates themselves with the hooks.
    immutable: bool = False

    # serialization stuff, only for analyses

    # file_cached: bool
    # # TODO: should the file caches results have a embeded version maybe?
    # encode_result: Callable[[Any], _Json]
    # decode_result: Callable[[_Json], Any]

    # Desperate attempt to make it work with doctests :/ not working
    @property
    def __doc__(self) -> str | None:
        return self.do_pass.__doc__
    @property
    def __name__(self) -> str:
        return self.name
    @property
    def __wrapped__(self) -> IPassFunction:
        return self.do_pass

    def __str__(self) -> str:
        # i.e. "Node analysis 'def_use_chains'" 
        return f'{self.runs_on.name.title()} {self.kind.name.lower()} {self.name!r}'
    
    @property
    def proto(self) -> PassPrototype:
        """
        Convenience to be able to call .proto on any PassLike.
        """
        return self

    def _replace(self, **kwargs: Any) -> PassPrototype:
        return attrs.evolve(self, **kwargs)
    
    def get_dependencies(self) -> Collection[PassLike]:
        """
        Get the direct dependencies of this pass.
        """
        return self.dependencies

    def get_all_dependencies(self) -> Collection[PassLike]:
        """
        Transitively iterate on all dependencies of this pass.
        """
        seen: set[PassLike] = OrderedSet()
        def _yield_deps(c: PassLike) -> Iterator[PassLike]:
            yield from (d for d in c.proto.get_dependencies() if d not in seen)
            yield from (d for d in chain.from_iterable(
                _yield_deps(dep) for dep in c.proto.get_dependencies()) if d not in seen)
        seen.update(_yield_deps(self))
        return seen

    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        return self._instanciate()(*args, **kwargs)
    
    # def proxy(self, level: _ElemKind) -> PassInstance:
    #     """
    #     Derive this pass to return new pass that results into a simple proxy that provide a C{get} method which trigers
    #     the original pass on the given node.
        
    #     This can be used to avoid calling repetitively ``passmanager.gather(pass, ...)``.
    #     """
    #     return self._instanciate().proxy(level)

    def missing_param(self) -> str | None:
        """
        Whether this pass prototype requires any parameter.
        """
        return self._instanciate().missing_param()
    
    def _instanciate(self) -> PassInstance:
        return PassInstance(self, FrozenDict())

    # Method to create a pattern from this pass.

    def like(self, **predicate: Callable[[object], bool]) -> PassPattern:
        """
        Create a pattern representing several possible derivations of 
        the pass to be matched against other passes. 

        Designed to be used for preserved analyses.

        @param predicate: The analysis parameters names to the match function. 
            A match function is a one-argument
            callable that returne whether the value for the parameter matches.
        """
        return PassPattern(self, **predicate)

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
    args: FrozenDict[str, Hashable]
    
    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        newpass = self
        if args or kwargs:
            newpass = newpass._add_args(*args, **kwargs)
        return newpass

    def _add_args(self, *args: Hashable, **kwargs: Hashable) -> PassInstance: 
        params = self.proto.params
        optional_params = self.proto.optional_params
        len_optinals = len(optional_params)
        len_required = len(params)

        if len(kwargs) > (len_optinals + len_required):
            raise TypeError(f'too many keyword parmeters, expected at most {len_optinals + len_required} keywords')
        if len(args) > len_required:
            raise TypeError(f'too many positional parmeters, expected at most {len_required} positionals and {len_required} keywords')
        
        self_args = self.args
        self_args_get = self_args.get
        args_dict = {}
        
        # support passing required parameters as positionals
        for pname, value in zip(params, args):
            # This prevents the creation of new instance of 
            # PassInstance whith the same params values.
            if value != (self_args_get(pname, _nah)):
                args_dict[pname] = value
        
        for pname, value in tuple(kwargs.items()):
            if (pname not in params) and (pname not in optional_params):
                raise TypeError(f'unexpected argument {pname!r}')
            if pname in args_dict:
                raise TypeError(f'got several values for parameter {pname!r}')
            if value != self_args_get(pname, _nah):
                args_dict[pname] = value
        
        if args_dict:
            return PassInstance(
                self.proto, 
                FrozenDict({**self.args, **args_dict})
            )
        else:
            return self

    # def proxy(self, level: _ElemKind) -> PassInstance:
    #     if self.proto.kind != _ANALYSIS:
    #         # Client need to write their own wrapper for transformation
    #         # to run it on all applicable nodes of a module for instance. This is 
    #         # a task that cannot be generalized for all tree types so it doesn't belong here.
    #         raise TypeError('cannot proxy a transformation')
    #     if (runs_on:=self.proto.runs_on) == _FOREST:
    #         raise ValueError('cannot proxy a forest analysis')
    #     if level < runs_on:
    #         raise ValueError('cannot proxy an analysis to a lower level')
    #     if level == runs_on:
    #         # that's a unssuported special case because... 
    #         raise ValueError('cannot proxy an analysis to the same level')
    #     if level == _FOREST:
    #         # To create the proxy we need to dynamically change it's prototype
    #         # in order to have the same name as the proxied analysis.
    #         new_proto = _forest_proxy_pass._replace(name=self.proto.name)
    #     elif level == _MTREE:
    #         new_proto = _mtree_proxy_pass._replace(name=self.proto.name)
    #     else:
    #         assert False
    #     return new_proto(proxied=self)
    
    # proxy.__doc__ = PassPrototype.proxy.__doc__ # yes this is supported by pydoctor

    def missing_param(self) -> str | None: 
        """
        Whether this pass instance is missing a required parameter.
        """
        if(any((missing:=p) not in self.args for p in self.proto.params)):
            return missing
        return None
    
    def _replace(self, **kwargs: Any) -> PassInstance:
        return attrs.evolve(self, **kwargs)

_posargs = frozenset((Parameter.POSITIONAL_OR_KEYWORD, 
                          Parameter.POSITIONAL_ONLY, ))
_unsupportedargs = frozenset((Parameter.VAR_KEYWORD, 
                          Parameter.VAR_POSITIONAL, ))

_runs_on_type_2_level = {Forest: _FOREST, MTree: _MTREE}

def new_pass_prototype(
        do_pass: Callable[..., CastableToDict], 
        *, 
        kind: _PassKind, # analysis or transformation ?
        on: type | Iterable[type], 
        dependencies: Sequence[PassLike] | None = None,
        cached: bool = True, # for analyses only
        immutable: bool = False, # for analyses only
        # file_cached: bool = False,
    ) -> PassPrototype: 
    """
    Create a pass prototype from a given callable and a handful of options.
    
    @param do_pass: A callable that contains the driving logic of your pass.
      By convention, the callable should be either
       
       - a generator function yielding tuples: (key, value)
       - a function returning a dict
      
      A passe can provide metadata that are not directly
      meant to be presented to the users but rather use to internally optimize runs.
    
      Support two variants: 
        
        - no parameter -> two posargs::
            def f(c, node): ...
        - with parameters -> two posargs and x keywords::
            def f(c, node, *, arg1, arg2=False): ...
    
    @param kind: L{ANALYSIS} or L{TRANSFORMATION}.
    @param on: The type of the element the pass is supposed to be run on. 
        I.e. L{Forest}, L{MTree}, L{ast.Module}, L{ast.FunctionDef}. 
        This can also be a tuple of types, but this is only applicable 
        if your pass runs on syntax tree nodes (not MTree or Forest). 
    @param dependencies: Sequence of the pass-like dependencies of this pass.
    @param cached: Whether this analysis should be kept in the cache. True by default.
    @param immutable: Wether this analysis will always return the same results.
        Applicable only if cache=True.
    @param longlived: Wether this analysis should always be preserved by all
        subsequent transformations. If it's not preserved, it will raise an exception.
        Applicable only if cache=True and immutable=False.
    """
    try:
        do_pass_sig = signature(do_pass)
    except Exception as e:
        raise TypeError('This functions is not supported at the moment '
                        f'because it can\'t be inspected: {do_pass}') from e
    
    # Validate the signature and extract parameters
    params: list[str] = [] # names
    optional_params: dict[str, Hashable] = {} # names to their defaults
    
    nb_pos_args = 0
    for param_name, param in do_pass_sig.parameters.items():
        if param.kind in _posargs: # we have a positional argument
            nb_pos_args += 1
            if nb_pos_args > 2:
                raise TypeError(
                    'A pass function must not take more '
                    'than two positional arguments, please use keyword-only '
                    'arguments to delcare your pass parametes.')
        elif param.kind in _unsupportedargs:
            raise TypeError('A pass function must not use variable length arguments, '
                            'please use keyword-only arguments')
        
        # It's keyword-only argument, so record the param name
        # with the default value if it has been given.
        elif (default:=param.default) is Parameter.empty:
            params.append(param_name)
        else:
            optional_params[param_name] = default
    
    if nb_pos_args != 2:
        raise TypeError(
                    'A pass function must take exactly '
                    f'tow positional arguments ({nb_pos_args} detected).')

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
    runs_on_level = _runs_on_type_2_level.get(runs_on_type, _NODE) # type: ignore[arg-type]

    proto =  PassPrototype(
        do_pass, 
        name=do_pass.__name__, 
        kind=kind,
        runs_on=runs_on_level, 
        runs_on_type=runs_on_type,
        params=params,                      # type: ignore[arg-type] 
        optional_params=optional_params,
        dependencies=dependencies or (),    # type: ignore[arg-type] 
        cached=cached,
        immutable=immutable,
    )
    
    return proto

def _pass_decorator(**kwargs: Any) -> Callable[[Callable], PassPrototype]:
    """
    Wraps L{new_pass_prototype} to be used as a decorator.
    """
    def decorator(function: Callable) -> PassPrototype:
        return new_pass_prototype(function, **kwargs)
    return decorator

transformation = partial(_pass_decorator, kind=_TRANSFORMATION)
"""
Main decorators to create a transformation
"""

analysis = partial(_pass_decorator, kind=_ANALYSIS)
"""
Main decorators to create an analysis
"""

class _PassRunMetadata:
    """
    Encapsulate the data passed arround in between the context and the runner. 

    Currently this only includes the knowledge level of the pass.
    """
    __slots__ = 'knowledge',

    def __init__(self) -> None:

        self.knowledge: int = 0

class PassContext:
    """
    Class that does the book-keeping of the chains of passes runs.

    The context tracks which pass requires what "level of knownledge". 
    This concept is closely related to the "pointer" concept in the sens that
    the knowledge can only be one of these three values: forest, mtree or node.
    """

    __slots__ = '_knowledge_stack',

    # maintains a "stack" of running passes and on which element
    # the stack is implemented as dict because it stores the knowledge
    # level of the pass run as well.
    def __init__(self) -> None:
        self._knowledge_stack: dict[_PassRun, int] = {}
    
    @property
    def _current_passrun(self) -> _PassRun:
        try:
            return next(reversed(self._knowledge_stack))
        except StopIteration:
            raise RuntimeError('no pass is currently running')

    @contextmanager
    def _push_pass(self, passe: PassInstance, pointer: _Pointer) -> Iterator[_PassRunMetadata]:
        
        key: _PassRun = (passe, pointer)
        
        if key in self._knowledge_stack:
            # TODO: Use a exception subclass in order to potentially catch and
            # use another pass instead to break the cycle.
            raise RuntimeError(f'cycle detected with pass: {key}')

                                     # cast it to int to internalize the int, 
                                     # TODO: is this necessary?
        self._knowledge_stack[key] = int(passe.proto.runs_on)
       
        # TODO: Might be interesting to optimize the remove mtree transformation: 
        # Yield a context tracker that is able to say which knowledge 
        # the pass accessed as well as the complete list of dependent mtrees 
        # in the case of a forest knowledge analysis.
        # We can do this safely only if no direct access to the forest is done.
        # Forest proxies can be used to gather info for a different module but 
        # the pass should never directly read the content of the Forest. 
        meta = _PassRunMetadata()

        # enter context managed code
        yield meta
        
        # This is just in case someone does something stupid
        if __debug__:
            e = next(reversed(self._knowledge_stack))
            if e is not key:
                raise RuntimeError(f'pass context is confused: {e} is not {key}')
            del e

        # pop element from "stack" and record knowledge level in 'meta'.
        meta.knowledge = pass_run_knowledge = self._knowledge_stack[key]
        del self._knowledge_stack[key]

        # so at this time pass_run_knowledge contains the maximum runs_on level of the "passe"
        # and all it's used dependencies.
        
        # propagate the knowledge of the dependency pass towards the calling pass if any.
        if self._knowledge_stack:
            curr = self._current_passrun
            if self._knowledge_stack[curr] < pass_run_knowledge:
                self._knowledge_stack[curr] = pass_run_knowledge

CACHE_KEYS = OrderedSet(
    (
     'passe', # PassInstance

     'knowledge', # integer: FOREST / MTREE / NODE
     'completeness',  # boolean
     
     'mtree', # MTree or None
     'node', # AnyNode or None
     )
)

@attrs.frozen(slots=True)
class CompletedPass:
    """
    The object that is returned from L{PassManager.run} function.
    """
    passe: PassInstance
    pointer: tuple[()] | tuple[MTree] | tuple[MTree, Any]
    result: Any
    completeness: bool
    update: bool
    preserved: Container[PassInstance]
    knowledge: int

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
    def __getattribute__(self, name: str) -> Any:
        # re-implement part of the descriptor protocol such that it
        # works dynamically at class instances level; see prepare().
        attr = super().__getattribute__(name)
        if isinstance(attr, _PassDependencyDescriptor):
            return attr.callback()
        return attr

@attrs.frozen(slots=True)
class Connector:
    """
    Connector to the passmanager, from inside a pass function.
    This is what we get as the first argument of pass functions like::

        @analysis(on=AST)
        def stuff(connector: Connector, node): ...
    """
    deps: Dependencies #: Namespace containing the declared dependencies
    gather: Callable[..., Any] #: See L{PassManager.gather}
    apply: Callable[..., bool] #: See L{PassManager.apply}
    run: Callable[..., CompletedPass] #: See L{PassManager.run}

# TODO: These classes should be generic to avoid the ugly Any...
# typing this framework turns out more difficult that expected.
@attrs.frozen(slots=True)
class Runner:
    """
    The runner and subclasses do the heavy lifting... 
    """
    _passmanager: PassManager
    _passe: PassInstance
    _pointer: _Pointer

    @contextmanager
    def push(self) -> Iterator[_PassRunMetadata]:
        with self._passmanager._ctx._push_pass(self._passe, self._pointer) as meta:
            yield meta

    def prepare(self) -> Connector:
        """
        Prepare the pass connector namespace before running a pass.
        """
        
        p = self._passe
        passe_proto = p.proto
        element: _SimplePointer = self._pointer[1:]
        pm = self._passmanager

        # validate the runtime type if running on nodes
        if passe_proto.runs_on == _NODE:
            if not element:
                raise AssertionError
            runs_on_type = passe_proto.runs_on_type
            if not isinstance(element[-1], runs_on_type):
                # This can happen when defining a pass that only runs on ast.Module and then calling
                # it from a pass that run on a child node, expecting the framework to understand that the dependent
                # analysis should run on the enclosing Module. It doesn't work like that at the moment; since
                # the passmanager doesn't understand the hierarchy in between ast.Module and, ast.BinOp, let's say.
                # If your pass requires to run on the root of the parse tree, use on=passmanager.MTree 
                # and access the module instance with '.root' attribute.
                raise TypeError(f'unexpected type, got {element[-1]!r}, should be of type {runs_on_type!r}')

        # Apply all transformations eagerly, since we use a descriptor for all analyses
        # we need to transitivsely iterate dependent tranforms and apply then now.
        for _t in passe_proto.get_all_dependencies():
            t_proto = _t if isinstance(_t, PassPrototype) else _t.proto
            
            if t_proto.kind != _PassKind.TRANSFORMATION:
                continue
            if (t_runs_on:=t_proto.runs_on) < (p_runs_on:=passe_proto.runs_on):
                # Since a NODE pass can be run on a MTree, we should explicitely
                # accept if we hit this case. BUT what if the transformation only 
                # applies to FunctionDef for instance ? well.. then it will fail
                # at the apply() stage with a TypeError. 
                # TODO: we might be able to check the config to see if pass can be run
                # on root nodes or not; but this will increase complexity.
                if t_runs_on == _NODE and p_runs_on == _MTREE:
                    apply_on_element: _SimplePointer = element

                else:
                    # it's not clear from the code but that's the only 
                    # possible value for this variable at this point. 
                    if __debug__: 
                        assert p_runs_on == _FOREST
                    # This is a limitation, but it's easy to write a simple wrapper
                    # on client side.
                    raise TypeError(f'{p} cannot depend - even transitively - on {_t}. '
                                    'A pass can only depend on transformations '
                                    'that runs on a compatible or enclosing level.')
            elif t_runs_on > p_runs_on:
                # the dependency runs on a upper scope level, trim what's required 
                lvldiff = t_runs_on - p_runs_on
                apply_on_element = element[:-lvldiff] # type:ignore[assignment]
            else:
                # same level
                apply_on_element = element
            
            pm.apply(_t, *apply_on_element)                    

        # create the analysis dependencies namespace.
        deps = Dependencies()
        for _a in passe_proto.get_dependencies():
            a_proto = _a.proto
            if a_proto.kind != _PassKind.ANALYSIS:
                continue
            if _missing:=_a.missing_param():
                # dependency is missing a required parameter and cannot be presented as a descriptor. 
                # Instead of trying to do something smart and complex, we fail early.

                raise TypeError(f'{p} cannot list {_a} in dependencies because it is missing '
                                f'a required parameter {_missing!r}')
               
                # So for instance a NODE analysis 'attribute' which require a 'name' parameter 
                # @analysis(on=ast.AST)
                # def my_pass(c, node):
                #   c.gather(attribute('some_name'), 'some_other_module_name')
                #   c.gather(attribute('some_name'), 'some_other_module_name', class_def)

            # TODO: More code should be shared with the first for loop up there...
            if (a_runs_on:=a_proto.runs_on) < (p_runs_on:=passe_proto.runs_on):
                # the dependency runs on a lower scope level
                if a_runs_on == _NODE and p_runs_on == _MTREE:
                    # a node analysis can implicitely be called on a Mtree, but
                    # other kinds of "promotions" are not supported.
                    dep_element: _SimplePointer = element
                else:
                    # this is true because we have only 3 levels of elements.
                    if __debug__: 
                        assert p_runs_on == _FOREST 
                    raise TypeError(f'{p} cannot list {_a} in dependencies. '
                                    'A pass can only depend on analyses '
                                    'that runs on a compatible or enclosing level.')
            elif a_runs_on > p_runs_on:
                # the dependency runs on a upper scope level, trim what's required 
                lvldiff = a_runs_on - p_runs_on
                dep_element = element[:-lvldiff] # type:ignore[assignment]
            else:
                # same level
                dep_element = element
            
            # the dependency can be converted to a descriptor
            # TODO: I'm sure there is a faster way to do it
            callback: Callable[[], Any] = partial(pm.gather, _a, *dep_element)
            setattr(deps, a_proto.name, _PassDependencyDescriptor(callback))
        
        # create the namespace
        if passe_proto.kind == _ANALYSIS:
            connector = Connector(
                deps = deps, 
                gather = pm.gather, 
                apply = None, # no apply() for analyses.
                run = pm.run, # but run() is unrestricted :/
            )
        else:
            connector = Connector(
                deps = deps, 
                gather = pm.gather, 
                apply = pm.apply,
                run = pm.run, 
            )

        return connector

    def do_pass(self, c: Connector) -> TransformationReturn | AnalysisReturn:
        # call the pass function
        p = self._passe
        _res = p.proto.do_pass(c, self._pointer[-1], **p.args)
        if not isinstance(_res, dict): 
            # cast the result to dict if it's a generator
            result: TransformationReturn | AnalysisReturn = dict(_res) # type:ignore[assignment]
        else:
            result = _res # type:ignore[assignment]
        # TODO: We might be able to use the indexer to avoid dummy iterations.
        # like hooks indexed on the pass kind and level
        # apply hooks
        if hooks:=self._passmanager.hooks:
            for h in hooks[_pass_kind_2_hook_kind[p.proto.kind]]:
                result = h(p, self._pointer, result) or result
        return result

    def run(self) -> CompletedPass:
        raise NotImplementedError()

@attrs.frozen(slots=True)
class PassManagerCache:
    """
    Wraps the generic L{Cache} to offer 
    helpers based on L{CompletedPass}.
    """
    _cache: Cache[_CacheKeyT, CompletedPass]

    def get(self, passe: PassInstance, pointer) -> CompletedPass | None:
        for k in self._mk_cache_keys_to_get_result(passe, pointer):
            if result:=self._cache.get(k): 
                # the result is cached :)
                return result
        return None
    
    def set(self, result: CompletedPass):
        key = self._mk_cache_key_to_set_result(result)
        self._cache.set(key, result)
    
    def remove(self, key: _CacheKeyT) -> None:
        self._cache.remove(key)
    
    def search(self, **key) -> Collection:
        return self._cache.search(**key)

    def allkeys(self) -> Collection:
        return self._cache.allkeys()

    @staticmethod
    def _mk_cache_key_to_set_result(result: CompletedPass) -> _CacheKeyT:
        pointer: _Pointer = result.pointer
        # we do not use the forest part of the pointer here
        path = pointer + (None, ) * (3 - len(pointer))
        completeness = True
        return result.passe, result.knowledge, completeness, path[1], path[2]

    @staticmethod
    def _mk_cache_keys_to_get_result(passe:PassInstance, 
                                    pointer: _Pointer, ) -> Iterator[_CacheKeyT]:     
        # we do not use the forest part of the pointer here
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

@attrs.frozen(slots=True)
class AnalysisRunner(Runner):
    
    def run(self) -> CompletedPass:
        passe = self._passe
        pointer = self._pointer
        cache = self._passmanager.cache
        
        with self.push() as meta:
            if passe.proto.cached:
                # Try to fetch value from cache
                if result:=cache.get(passe, pointer): 
                    # The result is cached :)
                    return result

            # TODO: More code should be shared with TransformationRunner
            # run the analysis 
            ret: AnalysisReturn = self.do_pass(self.prepare()) # type: ignore[assignment]

        # by default all forest knowledge analyses are incomplete and other are complete. 
        # TODO: We currently do not validate if a tree or node analysis is ever marked as incomplete.
        #   in which case that would be an error of the developers.   
        knowledge = meta.knowledge
        ret.setdefault('completeness', knowledge != _FOREST)
        
        result = CompletedPass(
            passe, pointer[1:], 
            result=ret['result'], 
            completeness=ret['completeness'], 
            update=False, knowledge=knowledge, preserved=())
        
        if passe.proto.cached:
            # Set the analysis result in the cache once we have left the with: block.
            cache.set(result)
        
        return result

@attrs.frozen(slots=True)
class TransformationRunner(Runner):
    
    def run(self) -> CompletedPass:
        
        pointer = self._pointer
        passe = self._passe
        runs_on = passe.proto.runs_on
        
        with self.push() as meta:
            ret: TransformationReturn =  self.do_pass(self.prepare()) # type:ignore[assignment]
        
        if 'preserved' in ret:
            # process preserved analyses so instances are 
            # in a fast track compared to patterns, because the way 
            # it works patterns must be checked with __eq__
            # and pass instance can be checking with __hash__ making 
            # it much more efficient, so leverage this by using ChainSet.
            pure_passes: list[PassInstance] = []
            patterns: list[PassPattern] = []
            for a in ret['preserved']:
                if isinstance(a, PassPrototype):
                    pure_passes.append(a())
                elif isinstance(a, PassInstance):
                    pure_passes.append(a)
                else:
                    patterns.append(a)
            if patterns:
                preserved: Container[PassInstance] = ChainSet((frozenset(pure_passes), 
                                                           tuple(patterns)))
            else:
                preserved = frozenset(pure_passes)
        else:
            preserved = frozenset()
        
        knowledge = meta.knowledge       
        result = CompletedPass(
            passe, pointer[1:],
            update=ret['update'], 
            preserved=preserved, 
            completeness=False, result=None, 
            knowledge=knowledge, )
        
        if not ret['update']:
            # If the transformation did not affected the AST, return directly.
            # TODO: It would be good to cache this fact and not have to rerun the transform
            # again and again if we know it won't apply an update...
            return result
        
        cache = self._passmanager.cache
        cache_remove = cache.remove
        k1, k2, k3 = (), (), () #type: tuple[Iterable[_CacheKeyT], Iterable[_CacheKeyT], Iterable[_CacheKeyT]]

        # cached stuff needs to be invalidated, 
        # depending on the element type we're transforming.
        if runs_on == _FOREST:
            # This block is very special because FOREST transformations can only two kind of things:
            # an addition or a removal of a tree. 
            _proto = passe.proto
            
            if _proto is _remove_mtree:
                # Clears all forest knowledge analyses
                k1 = cache.search(knowledge=_FOREST)
                # Clears all analyses that are indexed in that module
                k2 = cache.search(mtree=self._passe.args['tree'])
            elif _proto is _add_mtree:
                # Clears all forest knowledge analyses that are not complete
                k1 = cache.search(knowledge=_FOREST, completeness=False)
            else:
                # otherwise it's a custom Forest transformation, we invalidate **everything**.
                k1 = cache.allkeys()
            
            # We can't possiblily know in advance which mtree 
            # a certain forest knowledge analysis will depend on when it will be ran.
            # the import graph does not carry all the information necessary to be certain 
            # an analysis might request modules that are not in the dependant of the current module. 
            # so we can't really cut down the number of cleared analyses because their module if not in the 
            # dependencies of the affected module here.   
        else:
            # mtree or node transformations
            tree = self._pointer[1] # type:ignore[misc]
            
            # - For a mtree transformation: 
            # - all forest knowledge analyses except few preserved
            # - all tree/node knowledge analyses belonging to a given module except few preserved
            k1 = cache.search(knowledge=_MTREE, mtree=tree)
            k2 = cache.search(knowledge=_NODE, mtree=tree)
            k3 = cache.search(knowledge=_FOREST)

        for key in itertools.chain(k1, k2, k3):
            analysis, *_ = key
            if analysis in preserved: 
                continue
            cache_remove(key)
        
        return result

_pass_kind_2_hook_kind: dict[_PassKind, Literal['transformation', 'analysis']] = {_ANALYSIS: 'analysis', _TRANSFORMATION: 'transformation'}

class PassManager: 

    def __init__(self, trees: Iterable[MTree] | None = None) -> None:
        
        # self.config = config # A config option could be used by the framework to provide better results. 
        # We might be able to increase our level of magical code by checking agains the type of the root node
        # or by traversing nodes with configured function, but this will increase again the complexity of code
        # that is already quite compex. Currently the core of the PassManager is library agnostic, and should stay.
        # class Config:
            # root_node_type = ast.Module
            # tree_attributes = ['filename', 'is_package', 'is_stub', 'code']
            # get_child_nodes = ast.iter_child_nodes

        self.trees = Forest(trees or [])
        self.cache = PassManagerCache(Cache(CACHE_KEYS, ['node']))
        self.hooks: dict[Literal['transformation', 'analysis'], list[Hook]] = defaultdict(list)
        
        self._ctx = PassContext()
        self._runners = {_ANALYSIS: AnalysisRunner, 
                         _TRANSFORMATION: TransformationRunner}

    def apply(self, transform: PassLike, *element: str | Element) -> bool:
        """
        High level method to run a tansformation.
        
        """
        if transform.proto.kind is not _TRANSFORMATION:
            raise TypeError
        return self.run(transform, *element).update

    def gather(self, analysis: PassLike, *element: str | Element) -> Any:
        """
        High level method to run an analysis.
        """
        if analysis.proto.kind is not _ANALYSIS:
            raise TypeError
        return self.run(analysis, *element).result

    def run(self, passe: PassLike, *element: str | Element) -> CompletedPass:
        """
        Lower level method to run any kind of pass and get a L{CompletedPass} instance in return.
        
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

        # create the pointer
        element = self._prepare_element(element, passe.proto.runs_on)
        pointer: _Pointer = (self.trees, ) + element # type: ignore
        runner = self._get_runner(passe(), pointer)
        return runner.run()
    
    # TODO: Would be good to be able to pass the tree attribute directly.   
    # @overload
    # def add(self, root: RootNode, identifier: str, **attributes: Hashable): ...
    # @overload
    # def add(self, tree: MTree): ...
    
    def add(self, tree: MTree) -> None:
        """
        Add a tree to the passmanager.
        """
        self.apply(_add_mtree(tree))
    
    def remove(self, tree: MTree) -> None:
        """
        Remove a tree from the passmanager,
        """
        self.apply(_remove_mtree(tree))

    def _prepare_element(self, element: tuple[str | Element,...], runs_on: _ElemKind) -> tuple[Element,...]:
        len_element = len(element)
        needs_to_append_root = False
        if runs_on == _NODE:
            if len_element == 1:
                # Very important for usability!!!
                # a NODE pass can be run on a MTree, 
                # in this case use the root module as the node.
                # This is only true if the pass can be run on 
                # the root no of the AST.
                needs_to_append_root = True
            elif len_element == 0:
                raise TypeError('a NODE pass expect at least one element argument (up to two), got 0')
        elif runs_on == _MTREE:
            if len_element != 1:
                raise TypeError(f'a MTREE pass expect exactly one element argument, got {len_element}')
        elif runs_on == _FOREST:
            if len_element != 0:
                raise TypeError(f'a FOREST pass expect exactly zero element argument, got {len_element}')
        
        if element:
            first_element = element[0]
            if not isinstance(first_element, MTree):
                # If the first element is not a mtree, 
                # try to fetch it from the forest, 
                # it can be either a identifier string
                # or the root node of the tree.
                module = self.trees[first_element]
                element = (module, ) + element[1:]
            elif first_element not in self.trees:
                # the tree is not in the system, so add it now.
                self.add(first_element)
            if needs_to_append_root:
                element += (element[0].root, )

        return element

    def _get_runner(self, passe: PassInstance, pointer: _Pointer) -> Runner:
        return self._runners[passe.proto.kind](self, passe, pointer)

# Internal builtin passes

# @analysis(on=Forest, cached=False)
# def _forest_proxy_pass(c: Connector, _: Forest, *, proxied: PassInstance) -> AnalysisReturn:
#     def inner_pass(*element: Element, **kwargs: Hashable) -> Any:
#         if kwargs:
#             runpass = proxied(**kwargs)
#         else:
#             runpass = proxied
#         return c.gather(runpass, *element)
#     return {'result': GetProxy(inner_pass), 'completeness': False}

# @analysis(on=MTree, cached=False)
# def _mtree_proxy_pass(c: Connector, node: MTree, *, proxied: PassInstance) -> AnalysisReturn:
#     def inner_pass(element: Element, **kwargs: Hashable) -> Any:
#         if kwargs:
#             runpass = proxied(**kwargs)
#         else:
#             runpass = proxied
#         return c.gather(runpass, node, element)
#     return {'result': GetProxy(inner_pass), 'completeness': False}

@transformation(on=Forest)
def _add_mtree(_: Connector, forest: Forest, *, tree: MTree) -> TransformationReturn:
    if tree in forest:
        return {'update': False}
    forest._add(tree)
    return {'update': True, 'preserved': []}

@transformation(on=Forest)
def _remove_mtree(_: Connector, forest: Forest, *, tree: MTree) -> TransformationReturn:
    if tree not in forest:
        return {'update': False, 'preserved': []}
    forest._remove(tree)
    return {'update': True, 'preserved': []}

# A analysis dependency is only a wrapper for calling gather
# with partial arguments already added.
# So the potential tables of dependency compatiblities matrix is something like

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

# A quick view of the usage

if __name__ == "__main__":

    pass

    # pm = PassManager()
    # assert isinstance(pm.trees, Forest)
    # pm.add(MTree('builtins', ast.parse(...)))
    # pm.add(MTree('typing', ast.parse(...)))
    # module = pm.trees['builtins']
    # for n in (n for n in ast.walk(module.root)):
    #    local_vars = pm.gather(local_variables, 'builtins', n)

   
