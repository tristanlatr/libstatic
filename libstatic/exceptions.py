import ast
import attr
import abc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import Scope

from .shared import ast_node_name


class StaticException(Exception, abc.ABC):
    """
    Base exception for the library.
    """
    
    node: object
    filename: Optional[str]

    def location(self) -> str:
        node = getattr(self.node, 'node', self.node)
        if not isinstance(node, ast.AST):
            if self.filename:
                return f"{self.filename}:?"
            return '?'
        lineno = getattr(node, "lineno", "?")
        col_offset = getattr(node, "col_offset", None)
        if col_offset:
            if self.filename:
                return f"{self.filename}:{lineno}:{col_offset}"
            else:
                return f"?:{lineno}:{col_offset}"
        else:
            if self.filename:
                return f"{self.filename}:{lineno}"
            else:
                return f"?:{lineno}"

@attr.s(auto_attribs=True)
class StaticNameError(StaticException):
    """
    Unbound name.
    """

    node: object
    filename: str

    def __str__(self) -> str:
        name = ast_node_name(self.node) if isinstance(self.node, ast.AST) else str(self.node)
        if name:
            return f"Unbound name {name!r}"
        else:
            return f"Unbound node {self.node}"


@attr.s(auto_attribs=True)
class StaticAttributeError(StaticException):
    """
    Attribute not found.
    """
    node: 'Scope'
    attr: str
    filename: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.location()}: Attribute {self.attr!r} not found in {self.node.name()!r}"


@attr.s(auto_attribs=True)
class StaticTypeError(StaticException):
    """
    A node in the syntax tree has an unexpected type.
    """

    node: object
    expected: str
    filename: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.location()}: Expected {self.expected}, got: {type(self.node)}"


@attr.s(auto_attribs=True)
class StaticImportError(StaticException):
    """
    An import target could not be found.
    """

    node: ast.alias
    filename: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.location()}: Import target not found: {ast_node_name(self.node)}"


@attr.s(auto_attribs=True)
class StaticValueError(StaticException):
    """
    Can't make sens of analyzed syntax tree.
    """

    node: ast.AST
    msg: str
    filename: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.location()}: Error, {self.msg}"


@attr.s(auto_attribs=True)
class StaticStateIncomplete(StaticException):
    """
    Missing required information about analyzed tree.
    Shouldn't be raised under normal usage of the library.
    """

    node: object
    msg: str

    def __str__(self) -> str:
        return f"Incomplete state, {self.msg}"


@attr.s(auto_attribs=True)
class StaticCodeUnsupported(StaticException):
    """
    Syntax is unsupported.
    """

    node: object
    msg: str
    filename: Optional[str]=None

    def __str__(self) -> str:
        return f"{self.location()}: Unsupported {self.msg}: {self.node}"


@attr.s(auto_attribs=True)
class StaticAmbiguity(StaticException):
    """
    Definition is ambiguous.
    """

    node: ast.AST
    msg: str

    def __str__(self) -> str:
        return f"Ambiguous definition, {ast_node_name(self.node)}: {self.msg}"


@attr.s(auto_attribs=True)
class StaticEvaluationError(StaticException):
    """
    The evaluation could not be completed.
    """

    node: ast.AST
    msg: str

    def __str__(self) -> str:
        return f"Evaluation error, {self.msg}"


@attr.s(auto_attribs=True)
class StaticUnknownValue(StaticException):
    """
    Used by literal eval when a used value is not known.
    """

    node: ast.AST
    msg: str

    def __str__(self) -> str:
        return f"Unkown value: {self.msg}"
