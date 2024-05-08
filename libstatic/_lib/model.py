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

from beniget.beniget import ordered_set # type: ignore

from .shared import ast_node_name
from .exceptions import NodeLocation, HasLocation

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

T = TypeVar("T", bound=ast.AST)


class _Msg(Protocol):
    def __call__(
        self, msg: str, ctx: Optional[HasLocation] = None, thresh: int = 0
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


class Type(Protocol):
    """
    The type of a Python expression.
    """
    
    @property
    def name(self) -> str:
        """
        The name of the type.

        For example, `Iterable` or `list`.
        """
    
    @property
    def qualname(self) -> str:
        """
        The full name of the type.

        For example, `typing.Iterable` or `builtins.list`.
        """
    
    @property
    def args(self) -> Sequence[Type]:
        """
        Arguments of a generic type if any.

        For example, ``(str, int)`` if the type is ``dict[str, int]``.
        """
    
    @property
    def annotation(self) -> str:
        """
        Express this type as a string.

        **The string might not be a valid Python expression**.

        For example, ``str | dict[str, Any]`` or ``(int) -> str`` for callables.
        """