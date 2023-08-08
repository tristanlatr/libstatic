import ast
import abc
import attr as attrs
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import Scope

from ._lib.shared import ast_node_name

@attrs.s(auto_attribs=True)
class StaticException(Exception, abc.ABC):
    """
    Base exception for the library.
    """
    
    node: object
    desrc: Optional[str] = None
    filename: Optional[str] = attrs.ib(kw_only=True, default=None)

    def location(self) -> str:
        node = getattr(self.node, 'node', self.node)
        if not isinstance(node, ast.AST):
            if self.filename:
                return f"{self.filename}:?"
            return '?'
        lineno = getattr(node, "lineno", "?")
        col_offset = getattr(node, "col_offset", None)
        nodecls = f'ast.{node.__class__.__name__}'
        if col_offset:
            if self.filename:
                return f"{nodecls} at {self.filename}:{lineno}:{col_offset}"
            else:
                return f"{nodecls} at ?:{lineno}:{col_offset}"
        else:
            if self.filename:
                return f"{nodecls} at {self.filename}:{lineno}"
            else:
                return f"{nodecls} at ?:{lineno}"

    @abc.abstractmethod
    def msg(self) -> str:
        ...

    def __str__(self) -> str:
        return f'{self.location()}: {self.msg()}'

@attrs.s
class StaticNameError(StaticException):
    """
    Unbound name.
    """
    desrc: None = attrs.ib(init=False, default=None)
    def msg(self) -> str:
        name = ast_node_name(self.node)
        if name:
            return f"Unbound name {name!r}"
        else:
            return f"Unbound node {self.node}"


@attrs.s
class StaticAttributeError(StaticException):
    """
    Attribute not found.
    """
    node: 'Scope'
    attr: str = attrs.ib(kw_only=True)
    desrc: None = attrs.ib(init=False, default=None)

    def msg(self) -> str:
        return f"Attribute {self.attr!r} not found in {self.node.name()!r}"


@attrs.s(auto_attribs=True)
class StaticTypeError(StaticException):
    """
    A node in the syntax tree has an unexpected type.
    """
    expected: str = attrs.ib(kw_only=True)
    desrc: None = attrs.ib(init=False, default=None)

    def msg(self) -> str:
        return f"Expected {self.expected}, got: {type(self.node).__name__}"


class StaticImportError(StaticException):
    """
    An import target could not be found.
    """
    node: ast.alias

    def msg(self) -> str:
        return f"Import target not found: {ast_node_name(self.node)}"


class StaticValueError(StaticException):
    """
    Can't make sens of analyzed syntax tree.
    """
    node: ast.AST

    def msg(self) -> str:
        return f"Error, {self.desrc}"


class StaticStateIncomplete(StaticException):
    """
    Missing required information about analyzed tree.
    Shouldn't be raised under normal usage of the library.
    """

    def msg(self) -> str:
        return f"Incomplete state, {self.desrc}"


class StaticCodeUnsupported(StaticException):
    """
    Syntax is unsupported.
    """

    def msg(self) -> str:
        return f"Unsupported {self.desrc}"


class StaticAmbiguity(StaticException):
    """
    Definition is ambiguous.
    """
    node: ast.AST

    def msg(self) -> str:
        return f"Ambiguous definition: {self.desrc}"


class StaticEvaluationError(StaticException):
    """
    The evaluation could not be completed.
    """
    node: ast.AST

    def msg(self) -> str:
        return f"Evaluation error, {self.desrc}"


class StaticUnknownValue(StaticException):
    """
    Used by literal eval when a used value is not known.
    """

    node: ast.AST

    def msg(self) -> str:
        return f"Unkown value: {self.desrc}"
