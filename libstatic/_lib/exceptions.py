from __future__ import annotations

import ast
import abc
import attr as attrs
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import Scope, Def, Type

from .shared import ast_node_name

@attrs.s(auto_attribs=True, kw_only=True, str=False, frozen=True)
class NodeLocation:
    filename:'str|None' = None
    nodecls:'str|None' = None
    lineno:'int|None' = None
    # older python version don't alwasy set the col_offset field so we can't 
    # use it to compare locations.
    col_offset:'int|None' = attrs.ib(default=None, eq=False)

    @classmethod
    def make(cls, thing:ast.AST|Def|Type|object, filename:'str|None'=None) -> 'NodeLocation':
        """
        :param thing: A definition or an ast node.
        """
        if thing.__class__.__name__ == 'Type' and hasattr(thing, 'get_meta'):
            loc = getattr(thing, 'get_meta')('location')
            if loc:
                return loc
        node = getattr(thing, 'node', thing)
        if not isinstance(node, ast.AST):
            return NodeLocation(filename=filename)
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        nodecls = f'ast.{node.__class__.__name__}'
        return NodeLocation(filename=filename, 
                            nodecls=nodecls, 
                            lineno=lineno, col_offset=col_offset)

    def __str__(self):
        if self.nodecls is None:
            if self.filename:
                return f"{self.filename}:?"
            return '?'
        lineno = self.lineno if self.lineno is not None else "?"
        if self.col_offset:
            if self.filename:
                return f"{self.nodecls} at {self.filename}:{lineno}:{self.col_offset}"
            else:
                return f"{self.nodecls} at ?:{lineno}:{self.col_offset}"
        else:
            if self.filename:
                return f"{self.nodecls} at {self.filename}:{lineno}"
            else:
                return f"{self.nodecls} at ?:{lineno}"
            
@attrs.s(auto_attribs=True)
class StaticException(Exception, abc.ABC):
    """
    Base exception for the library.
    """
    
    node: ast.AST|Def|Type|object
    desrc: Optional[str] = None
    filename: Optional[str] = attrs.ib(kw_only=True, default=None)

    def location(self) -> NodeLocation:
        return NodeLocation.make(self.node, self.filename)

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
    node:ast.Name|str
    desrc: None = attrs.ib(init=False, default=None)
    def msg(self) -> str:
        if isinstance(self.node, str):
            name = self.node
        else:
            name = ast_node_name(self.node)
        return f"Unbound name {name!r}"

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
    node: ast.AST|Type

    def msg(self) -> str:
        return f"Error, {self.desrc}"


class StaticStateIncomplete(StaticException):
    """
    Missing required information about analyzed tree.
    Shouldn't be raised under normal usage of the library.
    """
    node:object
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
