import ast
import attr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Scope

from .shared import ast_node_name


class StaticException(Exception):
    """
    Base exception for the library.
    """


@attr.s(auto_attribs=True)
class StaticNameError(StaticException):
    """
    Unbound name.
    """

    node: object

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

    def __str__(self) -> str:
        return f"Attribute {self.attr!r} not found in {self.node.name()!r}"


@attr.s(auto_attribs=True)
class StaticTypeError(StaticException):
    """
    A node in the syntax tree has an unexpected type.
    """

    node: object
    expected: str

    def __str__(self) -> str:
        return f"Expected {self.expected}, got: {type(self.node)}"


@attr.s(auto_attribs=True)
class StaticImportError(StaticException):
    """
    An import target could not be found.
    """

    node: ast.alias

    def __str__(self) -> str:
        return f"Import target not found: {ast_node_name(self.node)}"


@attr.s(auto_attribs=True)
class StaticValueError(StaticException):
    """
    Can't make sens of analyzed syntax tree.
    """

    node: ast.AST
    msg: str

    def __str__(self) -> str:
        return f"Error, {self.msg}"


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

    def __str__(self) -> str:
        return f"Unsupported {self.msg}: {self.node}"


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
