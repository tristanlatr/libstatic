"""
This module contains the def-use models, use to represent the code.
"""
from __future__ import annotations
import ast
import inspect
from itertools import chain
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    Union,
    TypeVar,
    Callable,
    cast,
    overload,
)

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore

from beniget.beniget import ordered_set  # type: ignore
import attr as attrs
from .shared import ast_node_name
from .exceptions import NodeLocation

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

T = TypeVar("T", bound=ast.AST)


class _Msg(Protocol):
    def __call__(
        self, msg: str, ctx: Optional[ast.AST] = None, thresh: int = 0
    ) -> None:
        ...

# TODO use frozen dataclasses
# since ordered_set is not hashable we'll have to use unsafe_hash=True, eq=False, repr=False
class Def:
    """
    Model a use or a definition, either named or unnamed, and its users.
    """
    __slots__ = 'node', '_users', 'islive'

    def __init__(self, node:ast.AST, islive:bool=True) -> None:
        self.node = node
        self.islive = islive
        self._users: MutableSet["Def"] = ordered_set()
        self._setup()
    
    def _setup(self) -> None:
        pass

    def add_user(self, node: "Def") -> None:
        assert isinstance(node, Def)
        self._users.add(node)

    def name(self) -> Optional[str]:
        """
        If the node associated to this Def has a name, returns this name.
        Otherwise returns None.
        """
        return None

    def users(self) -> Collection["Def"]:
        """
        The list of ast entity that holds a reference to this node.
        """
        return self._users
    
    def __repr__(self) -> str:
        clsname = self.__class__.__qualname__
        name = self.name()
        if name:
            return f"<{clsname}(name={name})>"
        else:
            nodeclsname = self.node.__class__.__name__
            return f"<{clsname}(node=<{nodeclsname}>)>"

    def __str__(self) -> str:
        return self._str({})

    def _str(self, nodes: Dict["Def", int]) -> str:
        if self in nodes:
            return "(#{})".format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return "{} -> ({})".format(
                self.name() or self.node.__class__.__name__,
                ", ".join(u._str(nodes.copy()) for u in self._users),
            )

class NameDef(Def):
    """
    Model the definition of a name (abstract).
    """

    node: Union[
        ast.Module,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Name,
        ast.arg,
        ast.alias,
        ast.Attribute,
    ]

    def name(self) -> str:
        assert not isinstance(self.node, ast.Module)
        return ast_node_name(self.node)

class Scope(Def):
    """
    Model a python scope (abstract).
    """

    def name(self) -> str:
        raise NotImplementedError()

class OpenScope(Scope):
    """
    Model a open scope (abstract).
    """
    node: Union[ast.Module, ast.ClassDef]

class ClosedScope(Scope):
    """
    Model a closed scope (abstract).
    Closed scope have <locals>.
    """

    node: Union[
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.GeneratorExp,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
    ]

class Lamb(ClosedScope):
    """
    Model the definition of a lambda function.
    """
    node: ast.Lambda

    def name(self) -> str:
        return f"<{type(self.node).__name__.lower()}>"

class Comp(ClosedScope):
    """
    Model the definition of a generator or comprehension.
    """
    node: Union[ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp]

    def name(self) -> str:
        return f"<{type(self.node).__name__.lower()}>"


class Mod(NameDef, OpenScope):
    """
    Model a module definition.
    """
    __slots__ = (*Def.__slots__, '_modname', 'is_package', '_filename')

    node: ast.Module
    def __init__(self, 
                 node: ast.Module, 
                 modname: str, 
                 is_package: bool = False, 
                 filename: Optional[str] = None) -> None:
        super().__init__(node)
        self._modname = modname
        self.is_package = is_package
        self._filename = filename

    def name(self) -> str:
        return self._modname

    def filename(self) -> str:
        return self._filename or self._modname


class Cls(NameDef, OpenScope):
    """
    Model a class definition.
    """
    node: ast.ClassDef

class Func(NameDef, ClosedScope):
    """
    Model a function definition.
    """

    node: Union[ast.FunctionDef, ast.AsyncFunctionDef]


class Var(NameDef):
    """
    Model a variable definition.
    """

    node: ast.Name

class Attr(NameDef):
    """
    Model an attribute definition.
    """
    node: ast.Attribute


class Arg(NameDef):
    """
    Model a function argument definition.

    This demonstrate the `Arg.to_parameter` method:
    
    >>> from libstatic import Project
    >>> node = ast.parse('def f(a:int, b:object=None, *, key:str, **kwargs):...')
    >>> p = Project()
    >>> _ = p.add_module(node, 'test')
    >>> p.analyze_project()
    >>> func_node = node.body[0]
    >>> args = (p.state.get_def(n) for n in ast.walk(func_node.args) if isinstance(n, ast.arg))
    >>> parameters = [a.to_parameter() for a in args]
    >>> sig = inspect.Signature(parameters)
    >>> str(sig)
    '(a:...Name..., b:...Name...=...Constant..., *, key:...Name..., **kwargs)'

    """
    __slots__ = (*Def.__slots__, 'default', 'kind')

    node: ast.arg
    def __init__(self, 
                 node: ast.arg, 
                 islive: bool, 
                 default: ast.expr | None, 
                 kind: inspect._ParameterKind) -> None:
        super().__init__(node, islive)
        self.default = default
        self.kind = kind
        """
        One of ``Parameter.POSITIONAL_ONLY``, ``Parameter.POSITIONAL_OR_KEYWORD``, 
        ``Parameter.VAR_POSITIONAL``, ``Parameter.KEYWORD_ONLY`` or ``Parameter.VAR_KEYWORD``.

        :see: `inspect.Parameter.kind`
        """
    
    def to_parameter(self) -> inspect.Parameter:
        """
        Cast this `Arg` instance into a `inspect.Parameter` instance.
        """
        return inspect.Parameter(self.node.arg, self.kind, 
            default=self.default or inspect.Parameter.empty, 
            annotation=self.node.annotation or inspect.Parameter.empty)


class Imp(NameDef):
    """
    Model an imported name definition.
    """
    __slots__ = (*Def.__slots__, 'orgmodule', 'orgname')
    
    node: ast.alias
    def __init__(self, 
                 node: ast.alias, 
                 islive: bool, 
                 orgmodule: str, 
                 orgname: Optional[str] = None) -> None:
        super().__init__(node, islive)
        self.orgmodule = orgmodule
        self.orgname = orgname

    def target(self) -> str:
        """
        Returns the qualified name of the the imported symbol.
        """
        if self.orgname:
            return f"{self.orgmodule}.{self.orgname}"
        else:
            return self.orgmodule

_T = TypeVar("_T")
class LazySeq(Sequence[_T]):
    """
    A lazy sequence makes an iterator look like an immutable sequence.
    """
    def __init__(self, iterable:Iterable[_T]) -> None:
        self._iterator = iter(iterable)
        self._values: List[_T] = []
    
    def _curr(self,) ->int:
        return len(self._values)-1
    
    def _consume_next(self) -> _T:
        val = next(self._iterator)
        self._values.append(val)
        return val
    
    def _consume_until(self, key:int) -> None:
        if key < 0:
            self._consume_all()
            return
        while self._curr() < key:
            try:
                self._consume_next()
            except StopIteration:
                break
    
    def _consume_all(self) -> None:
        while 1:
            try:
                self._consume_next()
            except StopIteration:
                break
    @overload
    def __getitem__(self, key:int) -> _T:
        ...
    @overload
    def __getitem__(self, key:slice) -> list[_T]:
        ...
    def __getitem__(self, key:int|slice) -> _T | list[_T]:
        if isinstance(key, int):
            self._consume_until(key)
        else:
            self._consume_all()
        return self._values[key]
    
    def __iter__(self) -> Iterator[_T]:
        yield from self._values
        while 1:
            try:
                yield self._consume_next()
            except StopIteration:
                break
    
    def __len__(self) -> int:
        self._consume_all()
        return len(self._values)

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
class ChainMap(Mapping['_KT', '_VT']):
    """Combine multiple mappings for sequential lookup.

    For example, to emulate Python's normal lookup sequence:

        import __builtin__
        pylookup = ChainMap((locals(), globals(), vars(__builtin__)))        
    """

    def __init__(self, maps:Sequence[Mapping[_KT, _VT]]) -> None:
        self._maps = maps

    def __getitem__(self, key:_KT) ->_VT:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __len__(self) -> int:
        return len(set().union(*self._maps))     # reuses stored hash values if possible

    def __iter__(self) -> Iterator[_KT]:
        d = {}
        for mapping in reversed(self._maps):
            d.update(dict.fromkeys(mapping))    # reuses stored hash values if possible
        return iter(d)

# This class has beeen adapted from the 'astypes' project.
@attrs.s(frozen=True, auto_attribs=True)
class Type:
    """
    The type of a Python expression.
    """
    
    name: str
    """The name of the type.

    For example, `Iterable` or `list`.
    """

    
    scope: str = ''
    """The scope where the type is defined. This is often a module, 
    but it migth be a class or a function in some cases.

    For example, `typing` if the type is `Iterable`.
    Empty string for built-ins or other special cases.
    """

    args: Sequence['Type'] = attrs.ib(factory=tuple, kw_only=True)
    """Arguments of a generic type if any.

    For example, `(str, int)` if the type is `dict[str, int]`.
    """
    
    if not TYPE_CHECKING:
        # mypy is not very smart with the converter option :/
        scope: str = attrs.ib(default='', converter=lambda v: v if v!='builtins' else '')
        args: Sequence['Type'] = attrs.ib(factory=tuple, converter=tuple, kw_only=True)
            
    location: NodeLocation = attrs.ib(factory=NodeLocation, kw_only=True, eq=False, repr=False)
    """
    The location of the node that defined this type.
    It's set to an unknown location by default so special 
    types can be created dynamically. 
    """

    # TODO:
    # property: is_class: whether the Type represents a class, i.e not a special form Union/Literal/etc...
    #   A Protocol is a normal class
    # property: bases, mro, subclasses
    
    # Special types:
    _UNION: ClassVar = (('typing', 'Union'), )
    _LITERAL: ClassVar = (('typing', 'Literal'), 
                          (('typing_extensions', 'Literal')), )
    _TYPE: ClassVar = (('typing', 'Type'), )
    _CALLABLE: ClassVar = (('typing', 'Callable'), )
    _MODULE: ClassVar = (('types', 'ModuleType'), )

    Any: ClassVar[Type]
    Union: ClassVar[Type]
    Literal: ClassVar[Type]
    Type: ClassVar[Type]
    Callable: ClassVar[Type]
    Module: ClassVar[Type]

    @property
    def qualname(self) -> str:
        """The full name of the type.

        For example, `typing.Iterable` or `list` (for builtins).
        """
        scope = self.scope
        if scope:
            return f"{scope}.{self.name}"
        else:
            return self.name

    @property
    def unknown(self) -> bool:
        """
        We can create Type with empty name.
        It is used to denote an unknown type.
        """
        return not self.name

    @property
    def is_union(self) -> bool:
        return (self.scope, self.name) in Type._UNION
    
    @property
    def is_type(self) -> bool:
        return (self.scope, self.name) in Type._TYPE
    
    @property
    def is_callable(self) -> bool:
        return (self.scope, self.name) in Type._CALLABLE
    
    @property
    def is_module(self) -> bool:
        return (self.scope, self.name) in Type._MODULE
    
    @property
    def is_literal(self) -> bool:
        """
        A literal type means it's literal values can be recovered with:
        
        >>> type = Type.Literal.add_args(args=[Type('"val"')])
        >>> ast.literal_eval(type.args[0].name)
        'val'
        """
        return (self.scope, self.name) in Type._LITERAL

    @cached_property
    def annotation(self) -> str:
        """Represent the type as a string suitable for type annotations.

        The string is a valid Python 3.10 expression.
        For example, `str | dict[str, Any]`.
        """
        if self.unknown:
            return 'Any'
        if self.is_union:
            return ' | '.join(arg.annotation for arg in self.args)
        if self.args:
            args = ', '.join(arg.annotation for arg in self.args)
            return f'{self.name}[{args}]'
        return self.name

    def merge(self, other: 'Type') -> 'Type':
        """Get a union of the two given types.

        If any of the types is unknown, the other is returned.
        When possible, the type is simplified. For instance, `int | int` will be
        simplified to just `int`.
        """
        if self.unknown:
            return other
        if other.unknown:
            return self
        if self.supertype_of(other):
            return self
        if other.supertype_of(self):
            return other

        # if one type is already union, extend it
        if self.is_union and other.is_union:
            return Type.Union.add_args(
                args=tuple(chain(self.args, other.args)),
            )
        if self.is_union:
            return self._replace(
                args=tuple(chain(self.args, (other,)))
            )
        if other.is_union:
            return other._replace(
                args=tuple(chain((self,), other.args))
            )

        # none goes last
        if self.name == 'None':
            args = (other, self)
        else:
            args = (self, other)
        return Type.Union.add_args(args=args)
    
    def _replace(self, **changes:str|Sequence[Type]|NodeLocation) -> Type:
        return attrs.evolve(self, **changes) # type:ignore

    def add_args(self, args: Iterable[Type]) -> Type:
        """Get a copy of the Type with the given args added in the list of args.
        """
        return self._replace(args=tuple(chain(self.args, args)))

    def supertype_of(self, other: 'Type') -> bool:
        # TODO: use a mapping of type-promotion instead of this.
        if self.name == 'float' and other.name == 'int':
            return True
        if self.name in ('Any', 'object'):
            return True
        if self.is_union:
            for arg in self.args:
                if arg.supertype_of(other):
                    return True

        # TODO Look superclasses
        if self.name != other.name:
            return False
        if self.scope != other.scope:
            return False
        if self.args != other.args:
            return False
        return True

Type.Any = Type('')
Type.Union = Type('Union', 'typing')
Type.Literal = Type('Literal', 'typing')
Type.Type = Type('Type', 'typing')
Type.Callable = Type('Callable', 'typing')
Type.Module = Type('ModuleType', 'types')

# class UnionType(Type):
#     ...

# class LiteralType(Type):
#     ...

# class FunctionLikeType(Type):
#     ...

# class CallableType(FunctionLikeType):
#     ...

# class OverloadedType(FunctionLikeType):
#     ...

# class AnyType(Type):
#     ...

# class OpenScopeType(Type):
#     ...

# class ModuleType(OpenScopeType):
#     ...

# class InstanceType(OpenScopeType):
#     ...

# class TypeType(OpenScopeType):
#     ...