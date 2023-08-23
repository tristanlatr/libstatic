from __future__ import annotations
from _ast import AST

import ast
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Iterator, TypeVar, Union, List, TYPE_CHECKING

from .._lib.model import Type, Scope
from .._lib.shared import node2dottedname

if TYPE_CHECKING:
    from .state import State

import attr

class _AnnotationStringParser(ast.NodeTransformer):
    """When given an expression, the node returned by L{ast.NodeVisitor.visit()}
    will also be an expression.
    If any string literal contained in the original expression is either
    invalid Python or not a singular expression, L{SyntaxError} is raised.
    """

    def _parse_string(self, value: str) -> ast.expr:
        statements = ast.parse(value).body
        if len(statements) != 1:
            raise SyntaxError("expected expression, found multiple statements")
        stmt, = statements
        if isinstance(stmt, ast.Expr):
            # Expression wrapped in an Expr statement.
            expr = self.visit(stmt.value)
            assert isinstance(expr, ast.expr), expr
            return expr
        else:
            raise SyntaxError("expected expression, found statement")

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        value = self.visit(node.value)
        if isinstance(value, ast.Name) and value.id == Type.LITERAL:
            # Literal[...] expression; don't unstring the arguments.
            slice = node.slice
        elif isinstance(value, ast.Attribute) and value.attr == Type.LITERAL:
            # typing.Literal[...] expression; don't unstring the arguments.
            slice = node.slice
        else:
            # Other subscript; unstring the slice.
            slice = self.visit(node.slice)
        return ast.copy_location(ast.Subscript(value, slice, node.ctx), node)

    # For Python >= 3.8:

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        value = node.value
        if isinstance(value, str):
            return ast.copy_location(self._parse_string(value), node)
        else:
            const = self.generic_visit(node)
            assert isinstance(const, ast.Constant), const
            return const

    # For Python < 3.8:

    def visit_Str(self, node: ast.Str) -> ast.expr:
        return ast.copy_location(self._parse_string(node.s), node)

def _union(*args:Union[Type, str]) -> Type:
    # For testing only
    new_args:List[Type] = []
    for arg in args:
        if isinstance(arg, str):
            arg = Type.new(arg)
        new_args.append(arg)
    return Type.new(Type.UNION, args=new_args)

class _AnnotationToType(ast.NodeVisitor):
    """
    Converts an annotation into a L{Type}.
    """

    def __init__(self, state:State, scope:Scope) -> None:
        self.state = state
        self.scope = scope
        self.in_literal = False

    def generic_visit(self, node: AST) -> Any:
        raise ValueError(f'unexcepted node in annotation: {node}')

    def visit(self, expr:ast.AST) -> Type:
        """
        Callers should catch any L{Exception}.
        """
        return super().visit(expr)

    def visit_Name(self, node: ast.Name) -> Type:
        qualname = self.state.expand_expr(node) or self.state.expand_name(
            self.scope, node.id)
        if qualname:
            module, _, name = qualname.rpartition('.')
            return Type.new(name, module=module)
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        else:
            # Unbound name in annotation :/
            # TODO: log a warning
            return Type.new(node.id)
    
    def visit_Attribute(self, node: ast.Attribute) -> Type:
        dottedname = node2dottedname(node)
        if not dottedname:
            # the annotation is something like func().Something, not an actual name.
            # inside an annotation, this generally does not mean anything special.
            # TODO: Leave a warning or raise.
            return Type.new(node.attr)
        
        qualname = self.state.expand_expr(node) or self.state.expand_name(
            self.scope, '.'.join(dottedname))
        if qualname:
            module, _, name = qualname.rpartition('.')
            return Type.new(name, module=module)
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        else:
            # TODO: Leave a warning, the name is unbound
            return Type.new(node.attr, module='.'.join(dottedname[:-1]))
    
    def visit_Subscript(self, node: ast.Subscript) -> Type:
        left = self.visit(node.value)
        if left.is_literal:
            self.in_literal = True
        
        if isinstance(node.slice, ast.Tuple):
            args = [self.visit(el) for el in node.slice.elts]
            left = left._replace(args=args)
        else:
            arg = self.visit(node.slice)
            if arg:
                left = left._replace(args=[arg])    
        # nested literal are considered invalid annotations
        if left.is_literal:
            self.in_literal = False
        return left

    def visit_BinOp(self, node: ast.BinOp) -> Type:
        # support new style unions
        if isinstance(node.op, ast.BitOr):
            left = self.visit(node.left)
            right = self.visit(node.right)
            return _union(left, right)
        else:
            raise ValueError(f"binary operation not supported: {node.op.__class__.__name__}")
    
    def visit_Ellipsis(self, _: Any) -> Type:
        return Type.new('...') 
    
    def visit_Constant(self, node: Union[ast.Constant, ast.Str, ast.NameConstant]) -> Type:
        if node.value is None:
            return Type.new('None')
        elif isinstance(node.value, type(...)):
            return self.visit_Ellipsis(None)
        if self.in_literal:
            return Type.new(repr(node.s if isinstance(node, ast.Str) else node.value))
        else:
            try:
                # unstring annotations as strings
                expr = _AnnotationStringParser().visit(node)
                if expr is node:
                    raise SyntaxError(f'unexpected {type(node.value).__name__}')
            except SyntaxError as e:
                raise ValueError('error in annotation') from e
            return self.visit(expr)
    
    visit_Str = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_List(self, node: ast.List) -> Type:
        # ast.List is used in Callable, but we do not fully support it at the moment.
        # TODO: are the lists supposed to only allowed in callables?
        return Type.new('', args=[self.visit(el) for el in node.elts])

class _TypeInference(ast.NodeVisitor):
    """
    Find the L{Type} of an expression.
    """

    def __init__(self, state:State) -> None:
        self.state = state
        # self.scope = scope

    def generic_visit(self, node: AST) -> Any:
        raise ValueError(f'unsupported node: {node}')
    
    def visit(self, expr:ast.AST) -> Type:
        """
        Callers should catch L{ValueError}.
        """
        return super().visit(expr)

    # AST expressions

    def visit_Constant(self, node: ast.Constant) -> Type:
        if node.value is None:
            return Type.new('None')
        return Type.new(type(node.value).__name__)
    
    def visit_JoinedStr(self, node: ast.JoinedStr) -> Type:
        return Type.new('str')
    
    # container types

    def visit_List(self, node: ast.List) -> Type:
        return Type.new('list')

    def visit_Tuple(self, node: ast.Tuple) -> Type:
        return Type.new('tuple')

    def visit_Set(self, node: ast.Set) -> Type:
        return Type.new('set')

    def visit_Dict(self, node: ast.Dict) -> Type:
        return Type.new('dict')
    
    # statements

    def visit_FunctionDef(self, node: ast.FunctionDef|ast.AsyncFunctionDef) -> Type:
        # TODO: get better at callables
        argstype = Type.new('')
        if node.returns is not None:
            returntype = _AnnotationToType(self.state, 
                            self.state.get_enclosing_scope(node)).visit(node.returns)
        elif isinstance(node, ast.AsyncFunctionDef):
            returntype = Type.new('Awaitable', module='typing',
                args=[Type.new('')])
        else:
            returntype = Type.new('')
        return Type.new('Callable', module='typing', args=[argstype, returntype])
    
    def visit_Module(self, node: ast.Module) -> Type:
        return Type.new('ModuleType', module='types')
    
    def visit_ClassDef(self, node: ast.ClassDef) -> Type:
        #TODO: this implementation ignores inner classes
        return Type.new('Type', module='typing', 
            args=[Type.new(node.name, 
                           module=self.state.get_qualname(
                            self.state.get_enclosing_scope(node)))])