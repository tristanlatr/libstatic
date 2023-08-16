"""
This module contains the def-use models, use to represent the code.
"""

import ast
from itertools import chain
import sys
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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
    Set,
    TextIO,
    Tuple,
    Union,
    Type,
    TypeVar,
    cast,
    overload,
)
import attr as attrs
from beniget.beniget import ordered_set  # type: ignore
from typeshed_client import get_stub_file, get_search_context
from typeshed_client.finder import parse_stub_file

from .shared import ast_node_name, node2dottedname
from .transform import Transform
from .._analyzer.asteval import LiteralValue, _LiteralEval, _GotoDefinition
from .exceptions import (
    StaticStateIncomplete,
    StaticNameError,
    StaticAttributeError,
    StaticImportError,
    StaticException,
    StaticValueError,
    StaticTypeError,
    StaticAmbiguity, 
    NodeLocation
)

if TYPE_CHECKING:
    from typing import Literal, NoReturn, Protocol
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
    r"""
    Model a class definition.

    This demonstrate the `Cls.ivars` attribute:

    >>> from libstatic import Project
    >>> p = Project()
    >>> m = p.add_module(ast.parse('class C:\n'
    ... ' def __init__(self, x):\n'
    ... '  self._x = x'), 'test')
    >>> p.analyze_project()
    >>> C, = p.state.get_local(m, 'C')
    >>> ivars = [f'{v.name()!r} {p.state.get_location(v)}' for v in C.ivars]
    >>> print('\n'.join(ivars))
    '_x' ast.Attribute at test:3:2
    """
    __slots__ = (*Def.__slots__, 'ivars')
    node: ast.ClassDef
    
    def _setup(self) -> None:
        super()._setup()
        self.ivars: Sequence['Attr'] = []
        """
        A sequence of `Attr` definitions to the ``self`` 
        parameter of methods in the class body.
        """

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
    """

    node: ast.arg


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
