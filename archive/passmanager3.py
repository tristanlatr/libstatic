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
    overload,
)

if TYPE_CHECKING:
    from typing import NotRequired, TypeAlias, TypedDict
else:
    final = lambda f: f
    TypedDict = object

from libstatic._lib.structures import (
    Cache,
    Indexer,
    FrozenDict,
    OrderedSet,
)

import attrs

# TODOs:
# - Rename 'Tree' to simply 'Tree'
# - Rename key 'preserved' to 'preserve'
# - Rename parameter 'dependencies' to 'deps'
# - Rename pamateter 'cached' to 'cache'
# - The Pointer type seems redundant since we actually
# don't need the forest to be passed arround
# because we ever operate on a single forest.
# - Seems like the differencve in between the pass 
# prototype and the pass instance is a bit blurry.
# maybe we could merge the two into a single class.

############ Typing related declarations

Element: TypeAlias = object
"""
Represent any element of the system: forest, tree, or any nodes,
a tuple, a string or any other builtins types are NOT elements.

An element must be weak referenciable.
"""
RootNode: TypeAlias = object
"Represent the root node of the module (typically ast.Module)"
Node: TypeAlias = object
"Represent any node in a module, including it's root node"

PassLike: TypeAlias = "PassPrototype | PassInstance"
CastableToDict: TypeAlias = "Iterable[tuple[str, Any]] | dict[str, Any]"
"""
Anything that can be casted to dict. 
"""

"""
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

    def __call__(
        self, c: Connector, element: Element, **kwargs: Hashable
    ) -> CastableToDict: ...


class AnalysisReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{analysis} function.
    """
    result: Any
    completeness: NotRequired[bool]

AnalysisReturnIter: TypeAlias = '''Iterator[
    tuple[Literal['result'], Any] | 
    tuple[Literal['completeness'], bool]]'''

AnalysisReturn: TypeAlias = 'AnalysisReturnMap | AnalysisReturnIter'

class TransformationReturnMap(TypedDict):
    """
    The expected strucutre of the mapping-ish (it can a generator of key-values pairs)
    returned from a B{transformation} function.
    """
    update: bool
    preserved: NotRequired[Iterable[PassLike | PassPattern]]

TransformationReturnIter: TypeAlias = '''Iterator[
    tuple[Literal['update'], bool] | 
    tuple[Literal['preserved'], Iterable[PassLike | PassPattern]]]'''

TransformationReturn: TypeAlias = 'TransformationReturnMap | TransformationReturnIter'

_ElementPath: TypeAlias = (
    "tuple[Forest,] | tuple[Forest, Tree] | tuple[Forest, Tree, Node]"
)
"""
The path of an element in the system under one of these forms: 
    
    - forest
    - forest, tree
    - forest, tree, node
"""
_SimpleElementPath: TypeAlias = "tuple[()] | tuple[Tree,] | tuple[Tree, Node]"
"""
What's left from the element path when the forest is trimmed.
"""

_PassRun: TypeAlias = "tuple[PassInstance, _ElementPath]"
"""
A "pass run" stores a pass and on which element it has been run.
"""

_CacheKeyT: TypeAlias = Tuple[
    "PassInstance", int, bool, "Hashable | None", "Hashable | None"
]


THookObj = TypeVar('THookObj', 'UnpreparedPass', 'PreparedPass', 'CompletedPass', 'FailedPass')
Hook: 'TypeAlias' = Callable[[THookObj], THookObj | None]
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
    - knowledge: 'node',  'tree',  'forest' - only for 'completed' hooks.
"""

class IPluginRegistrar(Protocol):
    hooks: Hooks
    def configure(passe: PassLike, name: str) -> None: ...

class IPluginFactory(Protocol):
    def __call__(self) -> IPlugin: ...

class IPlugin(Protocol):
    """
    An plugin class:
        - has a name
        - can support a variety of different parsers, use 'all' special word to indicate 
            a plugin supports all kind of parsers.
        - can provide gather()/apply()/run() keywords, aka run_keywords
        - can provide analysis()/transformation() keywords, aka pass_keywords
        - can provide passe yield points keywords, aka yield_keywords
        - can installs hooks
        - can configure a pass alias as string
        - register method MUST at least return self, 
            - Returned instance of the plugin will be stored in the PassManager locals 
              as a attribute of the given `name`. 
            - A plugin can register other plugins and yield their instances
              from the register() method such that theyr will be stored 
              in the PassManager locals as well.
        - if your plugin has some __init__ arguments, they must be passed preemptively
          via the IPlugin.bind() classmethod, since the PassManager only handled zero-argument
          callabled returning an instance of IPlugin.
    """
    @classmethod
    def bind(cls, *args: Any, **kwargs: Any) -> IPluginFactory:
        def bound(): return cls(*args, **kwargs)
        return bound
    
    name: str
    # parsers: Collection[str]
    
    run_keywords: Collection[str]
    pass_keywords: Collection[str]
    yield_keywords: Collection[str]

    def register(self, r: IPluginRegistrar) -> Iterable[IPlugin]:...

_T = TypeVar("_T")

############ Actual framework


class Tree:
    """
    Encapsulate a single parse tree.

    All trees are required to have an identifier.
    This should be the python module name.

    @note: This is a read-only datastructure. Don't try to mutate identifier,
        root or attributes.
    """

    __slots__ = "__root", "__identifier", "attributes"

    # TODO: It would be best to provide a lazy loader for the parse tree, since
    # we'd always have to traverse the whole directory structure (because trees  
    # can't be bested under other trees), we want to avoid
    # parsing ASTs of trees that won't even be used because we scan for 
    # the whole standard library for instance. 
    def __init__(self, root: RootNode, identifier: str, 
                 **attributes: Hashable) -> None:
        
        self.__root = root
        self.__identifier = identifier
        self.attributes: Mapping[str, Hashable] = FrozenDict(**attributes)
        """
        Optional hashable metadata regarding this tree. 
        """

    @property
    def root(self) -> RootNode:
        """
        The root node of the tree.

        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = Tree(astmod, 'builtins')
        >>> assert tree.root is astmod
        """
        return self.__root

    @property
    def identifier(self) -> str:
        """
        The identifier of the module.

        >>> import ast
        >>> astmod = ast.parse('...')
        >>> tree = Tree(astmod, 'builtins')
        >>> assert tree.identifier == 'builtins'
        >>> assert tree == Tree(astmod, 'builtins')
        >>> assert tree != Tree(ast.parse('...'), 'builtins')
        """
        return self.__identifier
    
    def __str__(self):
        return f'Tree {self.__identifier!r}'

    def __repr__(self):
        return f'Tree({self.__root!r}, {self.__identifier!r}, **{self.attributes!r})'

    def __hash__(self) -> int:
        return hash((self.root, self.identifier, self.attributes))

    def __eq__(self, other: object) -> bool:
        if isinstance(self, Tree) and isinstance(other, Tree):
            return (
                self.root == other.root
                and self.identifier == other.identifier
                and self.attributes == other.attributes
            )
        return NotImplemented


class TreeNotFound(KeyError):
    """
    A subclass of KeyError that is used when a tree is not found in the forest.
    """


class Forest(Collection[Tree]):
    """
    A collection of trees.

    Provides a mapping-ish interface to access the trees.
    You should not initiate a forest yourself, the passmanager will do it.

    Values can be accessed both by module name or by module ast node.

    Mutation methods (add/remove) are private since these action should only
    be performed through the passmanager L{add}/L{remove} methods or by 
    the approriate forest-wide transformations: L{add_tree} or L{remove_tree}.

    >>> import ast
    >>> astmod = ast.parse('x = 1')
    >>> tree = Tree(astmod, 'mod1', filename='./mod1.py')
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

    __slots__ = "__identifier2tree", "__root2tree", "__trees"

    def __init__(self, trees: Iterable[Tree] | None = None) -> None:

        # each operation must maintain these 3 structures.
        self.__identifier2tree: dict[str, Tree] = {}
        self.__root2tree: dict[RootNode, Tree] = {}
        self.__trees: set[Tree] = set()

        if trees is not None:
            for t in trees:
                self._add(t)

    def _add(self, tree: Tree) -> None:
        # no-op is the tree is already in the collection.

        if tree in self.__trees:
            return

        if tree.identifier in self.__identifier2tree:
            raise ValueError(
                f"identifier {tree.identifier!r} "
                f"is already taken: {self[tree.identifier]}"
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

    def _remove(self, tree: Tree) -> None:
        if tree not in self:
            raise ValueError(f"tree not in the collection: {tree}")

        # remove the tree from the collection
        del self.__identifier2tree[tree.identifier]
        del self.__root2tree[tree.root]
        self.__trees.discard(tree)

    #  getitem interface

    def __getitem__(self, __key: str | RootNode) -> Tree:
        try:
            if isinstance(__key, str):
                return self.__identifier2tree[__key]
            else:
                return self.__root2tree[__key]
        except KeyError as e:
            raise TreeNotFound(__key) from e

    @overload
    def get(self, key: str | RootNode) -> Tree | None: ...
    @overload
    def get(self, key: str | RootNode, default: _T) -> Tree | _T: ...
    def get(self, key: str | RootNode, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # collection interface

    def __iter__(self) -> Iterator[Tree]:
        # we're not using __trees here because set is not ordered.
        # but dict is ordered. 
        return iter(self.__identifier2tree.values())

    def __len__(self) -> int:
        return len(self.__trees)

    def __contains__(self, other: object) -> bool:
        # A forest contains trees, root nodes and identifiers.
        return (
            other in self.__trees
            or other in self.__identifier2tree
            or other in self.__root2tree
        )

class Level(IntEnum):
    FOREST = 3
    TREE = 2
    NODE = 1
    
class PassKind(IntEnum):
    TRANSFORMATION = 10
    ANALYSIS = 11

class Trigger(IntEnum):
    UNPREPARED = 24
    PREPARED = 23
    COMPLETED = 22
    FAILED = 21

class _HookedAll(IntEnum):
    # for Hooks.install() method
    ALL = 99
class _HookedNotApplicable(IntEnum):
    # for Hooks.install() method
    NA = 0

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

    >>> # this pattern will never match any pass instance
    >>> like_none = count_objs.like(type=(lambda _: False)) 
    >>> # this pattern will match any instances of the pass 'count_objs'
    >>> like_any = count_objs.like(type=(lambda _: True))  
    >>> # this one idem 
    >>> like_any2 = count_objs.like()                       
    >>> # this one will match only if type=ast.ClassDef.
    >>> like_cls = count_objs.like(type=(lambda v: v is ast.ClassDef))

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
    __slots__ = "_match", "_passe"

    def __init__(
        self, passe: PassPrototype, **predicate: Callable[[object], bool]
    ) -> None:
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

    __hash__ = None # This class is not hashable by nature.
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
    kind: PassKind
    #: on what kind of element this pass runs on? this is conceptual.
    runs_on: Level

    #: on what type of object this pass runs on?
    #: a isinstance check will be done and L{TypeError}
    #: will be raised for any mismatch.
    runs_on_type: type | tuple[type, ...]

    # required parameters names declaration
    params: tuple[str, ...] = attrs.field(
        default=(), converter=tuple
    )  # at least an empty tuple

    # options names to their default values declaration
    optional_params: Mapping[str, Hashable] = attrs.field(
        default=FrozenDict(), converter=FrozenDict
    )  # at least an empty map

    # a sequence of dependecies that will be bound to 
    # variable inside the 'deps' of the connector.
    dependencies: Sequence[PassLike | str] = attrs.field(
        default=(), converter=tuple
    )  # at least an empty tuple

    keywords: Mapping[str, Any]

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
        return f"{self.runs_on.name.title()} {self.kind.name.lower()} {self.name!r}"

    @property
    def proto(self) -> PassPrototype:
        """
        Convenience to be able to call .proto on any PassLike.
        """
        return self

    def _replace(self, **kwargs: Any) -> PassPrototype:
        return attrs.evolve(self, **kwargs)

    def get_dependencies(self, get_passe: Callable[[str], PassLike]) -> Collection[PassLike]:
        """
        Get the direct dependencies of this pass.
        """
        return {get_passe(d) if isinstance(d, str) else d for d in self.dependencies}

    def get_all_dependencies(self, get_passe: Callable[[str], PassLike]) -> Collection[PassLike]:
        """
        Transitively iterate on all dependencies of this pass.
        """
        seen: set[PassLike] = OrderedSet()

        def _yield_deps(c: PassLike) -> Iterator[PassLike]:
            yield from (d for d in c.proto.get_dependencies(get_passe) if d not in seen)
            yield from (
                d
                for d in chain.from_iterable(
                    _yield_deps(dep) for dep in c.proto.get_dependencies(get_passe)
                )
                if d not in seen
            )

        seen.update(_yield_deps(self))
        return seen

    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        return self._instanciate()(*args, **kwargs)

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
            raise TypeError(
                "too many keyword parmeters, expected at most "
                f"{len_optinals + len_required} keywords"
            )
        if len(args) > len_required:
            raise TypeError(
                "too many positional parmeters, expected at most "
                f"{len_required} positionals and {len_required} keywords"
            )

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
                raise TypeError(f"unexpected argument {pname!r}")
            if pname in args_dict:
                raise TypeError(f"got several values for parameter {pname!r}")
            if value != self_args_get(pname, _nah):
                args_dict[pname] = value

        if args_dict:
            return PassInstance(self.proto, 
                                FrozenDict({**self.args, **args_dict}))
        else:
            return self

    def missing_param(self) -> str | None:
        """
        Whether this pass instance is missing a required parameter.
        """
        if any((missing := p) not in self.args for p in self.proto.params):
            return missing
        return None

    def _replace(self, **kwargs: Any) -> PassInstance:
        return attrs.evolve(self, **kwargs)


_posargs = frozenset(
    (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.POSITIONAL_ONLY,
    )
)
_unsupportedargs = frozenset(
    (
        Parameter.VAR_KEYWORD,
        Parameter.VAR_POSITIONAL,
    )
)

_on_cls_2_level = {Forest: Level.FOREST, Tree: Level.TREE}


def new_pass_prototype(
    do_pass: Callable[..., CastableToDict],
    *,
    kind: PassKind,
    on: type | Iterable[type],
    dependencies: Sequence[PassLike] | None = None,
    **kwargs: Any, 
    # memoize: bool = True,  # for analyses only
    # immutable: bool = False,  # for analyses only
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

      See L{AnalysisReturn} or L{TransformationReturn} 
      for supported keys and value types of the returned mapping.

      Support two variants:

        - no parameter -> two posargs::
            def f(c, node): ...
        - with parameters -> two posargs and x keywords::
            def f(c, node, *, arg1, arg2=False): ...

    @param kind: L{ANALYSIS} or L{TRANSFORMATION}.
    @param on: The type of the element the pass is supposed to be run on.
        I.e. L{Forest}, L{Tree}, L{ast.Module}, L{ast.FunctionDef}.
        This can also be a tuple of types, but this is only applicable
        if your pass runs on syntax tree nodes (not L{Tree} or L{Forest}).
    @param dependencies: Sequence of the pass-like dependencies of this pass.
   
    """
    try:
        do_pass_sig = signature(do_pass)
    except Exception as e:
        raise TypeError(
            "This functions is not supported at the moment "
            f"because it can't be inspected: {do_pass}"
        ) from e

    # Validate the signature and extract parameters
    params: list[str] = []  # names
    optional_params: dict[str, Hashable] = {}  # names to their defaults

    nb_pos_args = 0
    for param_name, param in do_pass_sig.parameters.items():
        if param.kind in _posargs:  # we have a positional argument
            nb_pos_args += 1
            if nb_pos_args > 2:
                raise TypeError(
                    "A pass function must not take more "
                    "than two positional arguments, please use keyword-only "
                    "arguments to delcare your pass parametes."
                )
        elif param.kind in _unsupportedargs:
            raise TypeError(
                "A pass function must not use variable length arguments, "
                "please use keyword-only arguments"
            )

        # It's keyword-only argument, so record the param name
        # with the default value if it has been given.
        elif (default := param.default) is Parameter.empty:
            params.append(param_name)
        else:
            optional_params[param_name] = default

    if nb_pos_args != 2:
        raise TypeError(
            "A pass function must take exactly "
            f"tow positional arguments ({nb_pos_args} detected)."
        )

    # Determine the runs_on_type based on provided 'on' param.
    runs_on_type = on
    if not isinstance(runs_on_type, type):
        if not isinstance(runs_on_type, tuple):
            runs_on_type = tuple(runs_on_type)
        if len(runs_on_type) == 0:
            raise ValueError('parameter "on" cannot be empty')
        if len(runs_on_type) != 1:
            # Validate the value since Tree and Forest should not be present in
            # passes that run on several nodes. 
            # This is a limitation that is necessary by design
            # since it is used used to differenciate a FOREST from a TREE pass, etc.
            if any((problematic := t) in _on_cls_2_level for t in runs_on_type):
                raise TypeError(
                    f"a pass cannot run both on {problematic} and on other types"
                )
        else:
            # only one value, so flatten it
            (runs_on_type,) = runs_on_type

    # Determine the runs_on_level
    runs_on_level = _on_cls_2_level.get(runs_on_type, Level.NODE)

    proto = PassPrototype(
        do_pass,
        name=do_pass.__name__,
        kind=kind,
        runs_on=runs_on_level,
        runs_on_type=runs_on_type,
        params=params,  # type: ignore[arg-type]
        optional_params=optional_params,
        dependencies=dependencies or (),  # type: ignore[arg-type]
        # cached=memoize,
        # immutable=immutable,
        keywords=kwargs, 
    )

    return proto


def _pass_decorator(**kwargs: Any) -> Callable[[Callable], PassPrototype]:
    """
    Wraps L{new_pass_prototype} to be used as a decorator.
    """

    def decorator(function: Callable) -> PassPrototype:
        return new_pass_prototype(function, **kwargs)

    return decorator


transformation = partial(_pass_decorator, kind=PassKind.TRANSFORMATION)
"""
Main decorators to create a transformation. 

"""

analysis = partial(_pass_decorator, kind=PassKind.ANALYSIS)
"""
Main decorators to create an analysis. 

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

class PreservedAnalyses:
    """
    Container for checking whether a given analyse is preserved after a
    transformatation has been applied. 
    """

    def __init__(self, get_passe: Callable[[str], PassLike], 
                 analyses: Iterable[PassPattern | PassLike | str] = ()):
        self.__passes: set[PassInstance] = set()

        # In order to to avoid linear time complexity based on the whole list
        # of patterns, we index the patterns based on the "pass prototype"
        # in a dict and matche only the subset coresponding.
        self.__patterns: dict[PassPrototype, list[PassPattern]] = {}
        self.__get_passe = get_passe
        for a in analyses:
            self.add(a)
    
    def add(self, pass_or_pattern: PassLike | PassPattern | str) -> None:
        if isinstance(pass_or_pattern, PassPrototype):
            self.__passes.add(pass_or_pattern())
        elif isinstance(pass_or_pattern, PassInstance):
            self.__passes.add(pass_or_pattern)
        elif isinstance(pass_or_pattern, str):
            self.add(self.__get_passe(pass_or_pattern))
        else:
            proto = pass_or_pattern._passe
            self.__patterns.setdefault(proto, []).append(pass_or_pattern)

    # because the manner the pattern matching is implemented (with __eq__)
    # it's not possible to simply remove a pattern element from a list
    # given an existing equivalent pattern instance. 
    # I choosed not to implement the remove() operation because it adds complexity, 
    # for someting without any usage at the moment.
    
    def __contains__(self, other: PassInstance) -> bool:
        return (other in self.__passes) or (
                (p:=other.proto) in self.__patterns 
                and other in self.__patterns[p])

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

    # @abc.abstractmethod
    # def maintain_cache(self, result: CompletedPass) -> None:
    #     ...
    
    # @abc.abstractmethod
    # def from_cache(self) -> CompletedPass | None:
    #     ...

@attrs.frozen(slots=True)
class AnalysisRunner(Runner[AnalysisReturnMap]):

    # def from_cache(self) -> CompletedPass | None:
    #     if self._passe.proto.cached:
    #         return self._passmanager.cache.get(self._passe, self._pointer)
    #     return None
    
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
    
    # def maintain_cache(self, result):
    #     if self._passe.proto.cached:
    #         # Set the analysis result in the cache once we have left the with: block.
    #         self._passmanager.cache.set(result)


# OLD CODE
    # def run(self) -> CompletedPass:
    #     passe = self._passe
    #     pointer = self._pointer
    #     cache = self._passmanager.cache

    #     with self.push() as meta:
    #         if passe.proto.cached:
    #             # Try to fetch value from cache
    #             if result := cache.get(passe, pointer):
    #                 # The result is cached :)
    #                 return result

    #         # TODO: More code should be shared with TransformationRunner
    #         # run the analysis
    #         ret: AnalysisReturnMap = self.do_pass(self.prepare())

    #     # by default all forest knowledge analyses are incomplete and other are complete.
    #     # TODO: We currently do not validate if a tree or 
    #     #   node analysis is ever marked as incomplete.
    #     #   in which case that would be an error of the developers.
    #     knowledge = meta.knowledge
    #     ret.setdefault("completeness", knowledge != _FOREST)

    #     result = CompletedPass(
    #         passe,
    #         pointer,
    #         result=ret["result"],
    #         completeness=ret["completeness"],
    #         update=False,
    #         knowledge=knowledge,
    #         preserved=(),
    #     )
    #     if passe.proto.cached:
    #         # Set the analysis result in the cache once we have left the with: block.
    #         cache.set(result)

    #     return result


@attrs.frozen(slots=True)
class TransformationRunner(Runner[TransformationReturnMap]):
    
    # def from_cache(self) -> CompletedPass | None:
    #     # transformation are not cached at all at present but 
    #     # in the future we might cache the fact that an transformation did not
    #     # updated
    #     return None
    
    # def maintain_cache(self, result: CompletedPass):
    #     if result.update:
    #         self._passmanager.cache.tracker.increment_rev(*result.pointer[1:])
    
    
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


# OLD CODE
    # def run(self) -> CompletedPass:
    #     with self.push() as meta:
    #         ret: TransformationReturnMap = self.do_pass(
    #             self.prepare()
    #         )  # type:ignore[assignment]


        # hooks should be applied here

        # if not ret["update"]:
        #     # If the transformation did not updated anything, return directly.
        #     # TODO: It would be good to cache this fact 
        #     # if we know it won't apply an update...
        #     return result

        # cache = self._passmanager.cache
        
        # to_remove: Iterable[_CacheKeyT] = ()

        # # cached stuff needs to be invalidated,
        # # depending on the element type we're transforming.
        # if runs_on == _FOREST:
        #     # FOREST transformations are special cased for these two kind of things:
        #     # - an addition of a tree.
        #     # - the removal of a tree. 
        #     # EDIT: This might be misguided after all, I suppose 
        #     # a better design would streamline all kind of transformation into
        #     # the same manner of handling post transformation cache maintenance.
        #     _proto = passe.proto
        #     if _proto is remove_tree:
        #         to_remove = chain(
        #             # Clears all forest knowledge analyses
        #             cache.search(knowledge=_FOREST), # o(1)
        #             # Clears all analyses that are indexed in that module
        #             cache.search(tree=self._passe.args["tree"])) # o(1)
        #     elif _proto is add_tree:
        #         # Clears all forest knowledge analyses that are not complete
        #         # We just use compleness=False, this implicitly implies that 
        #         # knowledge=_FOREST since only the forest-knowledge analysis might 
        #         # be incomplete. 
        #         to_remove = cache.search(completeness=False) # o(1)
        #     else:
        #         # Otherwise it's a custom Forest transformation that updated the contents
        #         # - it's important to say that it updated because forest-wide transformtions
        #         # that only calls c.apply(add_tree(...)) do not need to yield update=True
        #         # since it's goint o be handled by the add_tree transformation that is 
        #         # special cased up there, so these are fine. 
        #         #
        #         # We might be tempted to simply remove all results from the cache
        #         # because we don't know anything about this transformation, BUT that is
        #         # not a good idea since it could lead to terrible time complexity, which
        #         # would be against the point of the framework. 
        #         raise RuntimeError('unsupported forest-wide transformation')


        #     # We can't possiblily know in advance which tree
        #     # a certain forest knowledge analysis will depend on when it will be ran.
        #     # the import graph does not carry all the information necessary to be certain
        #     # an analysis might request modules that are not 
        #     # in the dependant of the current module.
        #     # so we can't really cut down the number of cleared 
        #     # analyses because their module is not in the
        #     # dependencies of the affected module here.

        #     # EDIT: But we could dynamically maintain the set of dependent modules used
        #     # to compute the analysis so that we known for sure which analysis required
        #     # which modules. In that case we don't need the import graph.

        #     # Also we can mark some analyses as strictly following the import graph
        #     # and outh not to access anything outside of module static dependencies.
        # else:
        #     # tree or node transformations
        #     tree = self._pointer[1]  # type:ignore[misc]

        #     # - For a tree or node transformation:
        #     # - all forest knowledge analyses except few preserved
        #     # - all tree/node knowledge analyses belonging to 
        #     # a given module except few preserved
        #     to_remove = chain(
        #         cache.search(tree=tree),  # o(1)
        #         cache.search(knowledge=_FOREST)) # o(1)

        #     # here for NODE transformation we should do an effort
        #     # and don't discard NODE analyses results of unrelated nodes in the same
        #     # module.

        # # What if the cost of iterating over 'to_remove' and checking whether 
        # # an anlysis is preserverved is higher than clearing eveything 
        # # and re-run the said analysis???
        # removed = set()
        # for key in to_remove: # o(number of analyses in scope)
        #     if key[0] in preserved: 
        #         # o(1) if no analysis patterns are used 
        #         # otherwise o(number of applicable patterns)
        #         continue
        #     if key in removed: # o(1)
        #         # Already removed, this can happen with the current setup since multiple
        #         # calls to cache.search() might return different collection containing
        #         # some of the same results.
        #         continue
        #     cache.remove(key)
        #     removed.add(key)
        # return resul

def _upper_if_string(v: object):
    if isinstance(v, str):
        return v.upper()
    return v

@attrs.frozen(slots=True)
class Hooks:
    """
    Container for the customization hooks. 
    The hooks are a manner to customize the process of running a pass.
    
    >>> my_cb = lambda o: print(o)
    >>> h = Hooks()
    >>> h.install(my_cb, when='after', kind='analysis', 
    ... level='tree', knowledge='all')
    >>> cbs = h.get(_AFTER, _ANALYSIS, _TREE, _TREE)
    >>> len(cbs)
    1
    >>> cbs[0](2)
    2
    """

    _hooks: Indexer[tuple[Hook, int, int, int, int]] = attrs.field(default_factory=lambda: Indexer(
        ['hook', 'when', 'kind', 'level', 'knowledge']))

    @staticmethod
    def _cast_string_values(when: str | int, 
                kind: str | int, 
                level: str | int, 
                knowledge: str | int) -> tuple[int, int, int, int]:
        # Cast everything to instances of integers.

        def _cast(v: str | int, maps: Iterable[type[IntEnum]]) -> int:
            if isinstance(v, int):
                return v
            for m in maps:
                try: return m[v]
                except KeyError: continue
            raise KeyError(v)
        
        return (
            _cast(when, [Trigger]), 
            _cast(kind, [PassKind, _HookedAll]),
            _cast(level, [Level, _HookedAll]), 
            _cast(knowledge, [_HookedNotApplicable, Level, _HookedAll])
        )
        
    def install(self, hook: Hook, *, 
        when: str | Trigger,
        kind: str | PassKind | _HookedAll,
        level: str | Level | _HookedAll,
        knowledge: str | Level | _HookedAll | _HookedNotApplicable = _HookedNotApplicable.NA, 
                ) -> None:  
        """
        Add a hook to the system.
        """
        # TODO: Use a priority-based order to apply hooks like we do for pydoctor's post-processing.
        # indtroduce the parameter priority. 

        when, kind, level, knowledge = self._cast_string_values(
            *map(_upper_if_string, [when, kind, level, knowledge]))

        # NA must always be used when the hooks runs before the pass, so validate that
        # the only trigger that run after the pass run is the COMPLETED.
        # TODO: Write nice error messages.
        if when != Trigger.COMPLETED: 
            if knowledge != _HookedNotApplicable.NA: 
                raise TypeError
        elif knowledge == _HookedNotApplicable.NA: 
            raise TypeError

        # Some combinaison of level/knowledge makes no sens: 
        # when the 'knowledge' is lower than the 'level'.
        if knowledge < level: 
            raise ValueError

        kinds = list(PassKind) if kind == _HookedAll.ALL else [kind]
        levels = list(Level) if level == _HookedAll.ALL else [level]
        knowledges = list(Level) if knowledge == _HookedAll.ALL else [knowledge]
        
        for combo in product(kinds, levels, knowledges):
            self._hooks.add((hook, when, *combo))

    def get(self, 
        when: Trigger,
        kind: PassKind,
        level: Level,
        knowledge: Level | _HookedNotApplicable.NA = _HookedNotApplicable.NA, 
                ) -> Iterable[Hook]:
        # This method doesn't support passing string values like install() for performance reason.
        return (k[0] for k in self._hooks.search(when=when, 
                                                 kind=kind, 
                                                 level=level, 
                                                 knowledge=knowledge))

    def uninstall(self, hook: Hook) -> None:
        hooks = self._hooks
        for k in hooks.search(hook=hook):
            hooks.discard(k)

CACHE_KEYS = OrderedSet(
    (   "passe",  # PassInstance
        "knowledge",  # integer: FOREST / TREE / NODE
        "completeness",  # boolean
        "tree",  # Tree or None
        "node",  # AnyNode or None
    ))

# TODO: The revision tracker should follow the import graph so we can preserve more
# forest-knowledge analyses. We can even automate this by using the used_paths combined 
# with a regular import analysis in order to detect wether the analyses followed the
# imports or not, and a flag can be added to the cache. This manner we can significantly
# reduce the overhead introduced by the multiplication of forest-knowledge analyses
class RevTracker:
    """
    The revision tracker is a part of the advanced caching features.

    Tracks the revision of all elements: the forest itself, trees and nodes.
    It MUST be informed with method `increment_rev` when a transformation occurs.
    Then it increments the relevant revisions hierachically in order to keep unrelated 
    nodes' revision unchanged. 

    One instance of this class can only be used to track revisions of elements 
    in single forest, it's also assumed that the tree instances are not shared across 
    multiple forests and all transformation are duly signaled to the instance of this class.
    """
    
    def __init__(self, get_anscestors: Callable[[Tree, Node], Iterable[Node]]):
        """
        :param get_anscestors: A callable that, given a tree and a 
            node that lives inside it, returns the ancestors nodes
            as a sequence from the root to the node's direct parent. 
        """
        self._forest_sentinel = object() # the forest object version
        self._get_anscestors = get_anscestors # dependency injection

        # We abuse the complex type as a two dimentional vector of ints.
        self._revisions: dict[Element, complex] = weakref.WeakKeyDictionary()
    
    def _elements(self, 
                  tree: Tree | None = None, 
                  node: Node | None = None) -> deque[object]:
        """
        Elements in this order: 

            - Node
            - Node ancestors up to the Module
            - Tree the node lives in
            - Forest (sentinel). 

        The returned deque will always at least contain the fores instance. 
        """
        elements = deque()
        if node:
            assert tree is not None
            # if there are nodes in between tree and node, add them to the stack
            elements.extendleft(self._get_anscestors(tree, node))
            elements.appendleft(node)
            # elements now contains the 
            # affected node first, then all it's ancestors.
        
        if tree:
            elements.append(tree)
        elements.append(self._forest_sentinel)
        return elements

    def increment_rev(self, 
                    tree: Tree | None = None, 
                    node: Node | None = None) -> None:
        """
        Signals that the given tree/node has been transformed 
        and increment it's revision hierachically. 
        
        It's illegal to give a node without a tree. 
        When no tree or node are given it wil be 
        interpreted as a forest transformation.
        """
        rev = self._revisions
        elements = self._elements(tree, node)
        
        def _incr(key: object, value: complex) -> None:
            rev.setdefault(key, 0j)
            rev[key] += value

        # the verion incrementation happends bottom-up, first we increment REAL part
        # of the directly affected element, then we increment the IMAGINARY part
        # of indirectly affected elements, up the global element version.
        _incr(elements.popleft(), 1)
        for e in elements:
            _incr(e, 1j)
    
    def rev(self, 
                 tree: Tree | None = None, 
                 node: Node | None = None, ) -> tuple[str, str, str]:
        """
        Computes the revision of the element at tree/.../node. 
        The revision is a three layered element.
        It's illegal to give a node without a tree. 
        When no tree or node are given the revision of the whole 
        forest will be returned.
        """
        rev = self._revisions
        elements = self._elements(tree, node)

        forest_rev = rev.get(elements.pop(), 0j)
        if not elements:
            return f"{forest_rev}", '0', '0'
        
        tree_rev = rev.get(elements.pop(), 0j)
        if not elements:
            return f"{forest_rev}", f"{tree_rev}", '0'
        
        # The version of a node is defined by the hash of it's own two dimentionned
        # vector (IMAGINARY and REAL) combined to the REAL dimention of the enclosing
        # elements. 
        node_rev = rev.get(elements.popleft(), 0j)
        parents_rev = (rev.get(e, 0j).real for e in elements)
        
        return f"{forest_rev}", f"{tree_rev}", '/'.join(map(str, chain(parents_rev, node_rev)))

# TODO: This protocol based appraoch is not suitable because we need to be able to write
# analyses that depends on the config, then the plugins can depend on the said
# analysis. 
# grosso-modo il nous faut pouvoir écrire
# @analysis(on=Tree)
# def ancestors(c, tree, *, config):
#     ...
#
# pm = PassManager(...)
# pm.configure(ancestors(get_children=...), 'ancestors')
# pm.gather('ancestors', tree, node)


def _import_graph_tracker(get_imports: Callable[[Tree], Iterable[str]]):...

@attrs.frozen(slots=True)
class PassManagerCache:
    """
    Wraps the generic L{Cache} class for caching L{CompletedPass} instances.
    """

    _cache: Cache[_CacheKeyT, CompletedPass]
    tracker: RevTracker

    def get(self, passe: PassInstance, pointer: _ElementPath) -> CompletedPass | None:
        for k in self._mk_cache_keys_to_get_result(passe, pointer):
            if result := self._cache.get(k):
                # the result is cached :)
                return result
        return None

    def set(self, result: CompletedPass):
        key = self._mk_cache_key_to_set_result(result)
        self._cache.set(key, result)

    @staticmethod
    def _mk_cache_key_to_set_result(result: CompletedPass) -> _CacheKeyT:
        pointer: _ElementPath = result.pointer
        # we do not use the forest part of the pointer here
        path = pointer + (None,) * (3 - len(pointer))
        completeness = True
        return result.passe, result.knowledge, completeness, path[1], path[2]

    @staticmethod
    def _mk_cache_keys_to_get_result(
        passe: PassInstance,
        pointer: _ElementPath,
    ) -> Iterator[_CacheKeyT]:
        # we do not use the forest part of the pointer here
        path = pointer + (None,) * (3 - len(pointer))
        p1 = path[1]
        p2 = path[2]
        runs_on = int(passe.proto.runs_on)

        yield passe, runs_on, True, p1, p2,
        while runs_on < Level.FOREST:
            runs_on += 1
            yield passe, runs_on, True, p1, p2,
        # The completeness can only be False for forest knowledge analyses.
        yield passe, runs_on, False, p1, p2,

    # TODO: There is too much boilerplate code around the cache management...
    def remove(self, key: _CacheKeyT) -> None:
        self._cache.remove(key)
    remove.__doc__ = Cache.remove.__doc__
    
    def search(self, **key) -> Collection:
        return self._cache.search(**key)
    search.__doc__ = Cache.search.__doc__
    
    def allkeys(self) -> Collection: # used for testing
        return self._cache.allkeys()
    allkeys.__doc__ = Cache.allkeys.__doc__

# What if the caching strategy could simply be a plugin ?

class NodeVisitor(object):
    """
    Like `ast.NodeVisitor`, but generic.
    """
    def __init__(self, get_children):
        self._get_children = get_children
    
    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for item in self._get_children(node):
            self.visit(item)

class _AncestorsVisitorMixin:
    def generic_visit(self, node):
        self._parents[node] = list(self._current)
        self._current.append(node)
        super().generic_visit(node)
        self._current.pop()

@analysis(on=Tree)
def _tree(_, tree: Tree): 
    yield 'result', tree

@analysis(on=Tree)
def _compute_ancestors(_, tree: Tree, *, get_children):
    class AncestorsVisitor(NodeVisitor, _AncestorsVisitorMixin):
        """
        Build the complete ancestor tree.
        """
        def __init__(self):
            super().__init__(get_children)
            self._parents = weakref.WeakKeyDictionary()
            self._current = []
    
    vis = AncestorsVisitor()
    vis.generic_visit(tree.root)
    yield 'result', vis._parents

@analysis(on=Node, deps=['ancestors'])
def _compute_ancestors_partial(c: Connector, node: Node, *, get_children):
    class AncestorsVisitor(NodeVisitor, _AncestorsVisitorMixin):
        """
        Build the partial ancestor tree.
        """
        def __init__(self):
            super().__init__(get_children)
            self._parents = {}
            self._current = c.deps.ancestors
    
    vis = AncestorsVisitor()
    vis.generic_visit(node)
    yield 'result',  vis._parents

@analysis(on=Node, deps=[_tree])
def _node_ancestors(c: Connector, node: Node, *, ancestors_per_tree):
    yield 'result', ancestors_per_tree[c.deps._tree][node]

class AncestorsViewPlugin(IPlugin):
    name = 'ancestors'
    parsers = ['all']
    pass_keywords = run_keywords = ()
    
    def __init__(self, get_children):
        self._get_children = get_children
        self._ancestors_per_tree: dict[Tree, dict[Node, Sequence[Node]]] = {}
    
    def register(self, r: IPluginRegistrar):
        r.configure(_compute_ancestors(get_children=self._get_children), '_compute_ancestors')
        r.configure(_compute_ancestors_partial(get_children=self._get_children), '_compute_ancestors_partial')

        self._main_analysis = main_analysis = _node_ancestors(ancestors_per_tree=self._ancestors_per_tree)
        r.configure(main_analysis, 'ancestors')

        install_completed_transform_hook = partial(r.hooks.install, 
                                              when=Trigger.COMPLETED, 
                                              kind=PassKind.TRANSFORMATION,
                                              knowledge='ALL')

        install_completed_transform_hook(
            self.on_completed_forest_transform, 
            level=Level.FOREST, 
        )

        install_completed_transform_hook(
            self.on_completed_tree_transform,
            level=Level.TREE,
        )

        install_completed_transform_hook(
            self.on_completed_node_transform,
            level=Level.NODE,
        )

    def on_completed_forest_transform(self, completed: CompletedPass) -> None:
        if not completed.update:
            return
        proto = completed.passe.proto
        tree = completed.passe.args['tree']
        if proto is add_tree:
            ans = completed.connector.gather('_compute_ancestors', tree)
            self._ancestors_per_tree[tree] = ans
        elif proto is remove_tree:
            del self._ancestors_per_tree[tree]
        else:
            raise TypeError(f'unsupported forest transform: {proto}')
    
    def on_completed_tree_transform(self, completed: CompletedPass) -> None:
        if not completed.update:
            return
        if self._main_analysis in completed.preserved:
            # no need to update the structure since the transformation should have done it.
            return
        tree = completed.pointer[1]
        ans = completed.connector.gather('_compute_ancestors', tree)
        self._ancestors_per_tree[tree] = ans

    def on_completed_node_transform(self, completed: CompletedPass) -> None:
        if not completed.update:
            return
        if self._main_analysis in completed.preserved:
            # no need to update the structure since the transformation should have done it.
            return
        tree = completed.pointer[1]
        node = completed.pointer[2]
        ans = completed.connector.gather('_compute_ancestors_partial', tree, node)
        self._ancestors_per_tree[tree].update(ans)
    
class ImportGraphViewPlugin(IPlugin):
    ...

class StronglyConnectedTrees(IPlugin):
    ...

class FillTreeAttrsPlugin(IPlugin):
    def __init__(self, tree_attrs: Sequence[tuple[str] | tuple[str, object]]):
        ...
    

import ast as __ast

@attrs.frozen()
class Config:
    parser: str # 'ast', 'gast', 'javalang', etc...
    plugins: Sequence[IPluginFactory]

def ast_config() -> Config:
    return Config(
            parser='ast',
            # get_children=__ast.iter_child_nodes, 
            # tree_attrs=[
            
            # ], 
            plugins=[
                FillTreeAttrsPlugin.bind(tree_attrs=[
                    ('filename', '<unknown>'),
                    ('is_package', False), 
                    ('is_namespace_package', False), 
                    ('is_stub', False), 
                    ('lines', None),
                ]), 

                AncestorsViewPlugin.bind(get_children=__ast.iter_child_nodes), 

            ])

default_config: Final = ast_config()

def _call(o: Callable[[], _T]) -> _T: 
    return o()

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
               tree: Tree) -> TransformationReturn:
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
                  tree: Tree | RootNode | str) -> TransformationReturn:
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

@transformation(on=Forest)
def pipeline(c, forest, passes: Sequence[PassLike]) -> TransformationReturn:
    """
    Group the given passes to run in a coherent order. 
    """
    # A pipeline is a forest transformation such that it emcompasses all other
    # analyses in tems of behavior. 


# Like in LLVM we could proxy function and classes passes to tree-wide passes
# with special passes called "proxy pass", but this more complexity on our side: 
# should it run on function and classes nested under others, I guess so... 
# but maybe tht behavior would be worth to be customizable.
# def function2tree(c, tree, passe):...
# def class2tree(c, tree, passe):...
# def tree2forest(c, forest, passe):...

# A analysis dependency is only a wrapper for calling gather
# with partial arguments already added.
# So the potential tables of dependency compatiblities matrix is something like

# NODE pass has NODE dep
# NODE pass has TREE dep
# NODE pass has FOREST dep
# TREE pass has TREE dep
# TREE pass has FOREST dep
# FOREST pass has FOREST dep

# A quick view of the usage

if __name__ == "__main__":

    pass

    # pm = PassManager()
    # assert isinstance(pm.trees, Forest)
    # pm.add(Tree('builtins', ast.parse(...)))
    # pm.add(Tree('typing', ast.parse(...)))
    # module = pm.trees['builtins']
    # for n in (n for n in ast.walk(module.root)):
    #    local_vars = pm.gather(local_variables, 'builtins', n)
