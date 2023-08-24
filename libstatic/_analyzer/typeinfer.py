from __future__ import annotations

import ast
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Iterator, Sequence, TypeVar, Union, List, TYPE_CHECKING

from .._lib.model import Type, Scope, Def, Cls
from .._lib.shared import node2dottedname
from .._lib.exceptions import StaticAmbiguity, StaticException, StaticNameError, StaticAttributeError, StaticValueError
from .asteval import _EvalBaseVisitor, _GotoDefinition

from beniget.beniget import BuiltinsSrc

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
            raise StaticValueError("expected expression, found multiple statements")
        stmt, = statements
        if isinstance(stmt, ast.Expr):
            # Expression wrapped in an Expr statement.
            expr = self.visit(stmt.value)
            assert isinstance(expr, ast.expr), expr
            return expr
        else:
            raise StaticValueError("expected expression, found statement")

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        value = self.visit(node.value)
        if isinstance(value, ast.Name) and value.id == 'Literal':
            # Literal[...] expression; don't unstring the arguments.
            slice = node.slice
        elif isinstance(value, ast.Attribute) and value.attr == 'Literal':
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
    new_args:tuple[Type, ...] = ()
    for arg in args:
        if isinstance(arg, str):
            arg = Type(arg)
        new_args += (arg,)
    return Type.Union.add_args(args=new_args)

class _AnnotationToType(ast.NodeVisitor):
    """
    Converts an annotation into a L{Type}.
    """

    # ISC License

    # Copyright (c) 2021, Timothée Mazzucotelli

    # Permission to use, copy, modify, and/or distribute this software for any
    # purpose with or without fee is hereby granted, provided that the above
    # copyright notice and this permission notice appear in all copies.

    # THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    # WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    # MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    # ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    # WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    # ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    # OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

    def __init__(self, state:State, scope:Scope) -> None:
        self.state = state
        self.scope = scope
        self.in_literal = False

    def generic_visit(self, node: ast.AST) -> Any:
        raise StaticValueError(f'unexcepted node in annotation: {node}')

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
            return Type(name, module)
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        elif node.id in BuiltinsSrc:
            # the builtin module might not be in the system
            return Type(node.id)
        else:
            # Unbound name in annotation :/
            # TODO: log a warning
            raise StaticNameError(node, filename=self.state.get_filename(node))
            
    
    def visit_Attribute(self, node: ast.Attribute) -> Type:
        dottedname = node2dottedname(node)
        if not dottedname:
            # the annotation is something like func().Something, not an actual name.
            # inside an annotation, this generally does not mean anything special.
            # TODO: Leave a warning or raise.
            raise StaticValueError(node, desrc='illegal expression in annotation', 
                                   filename=self.state.get_filename(node))
            return Type(node.attr)
        
        qualname = self.state.expand_expr(node) or self.state.expand_name(
            self.scope, '.'.join(dottedname))
        if qualname:
            module, _, name = qualname.rpartition('.')
            return Type(name, module)
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        else:
            # TODO: Leave a warning, the name is unbound
            raise StaticAttributeError(node, attr=node.attr)
            return Type(node.attr, '.'.join(dottedname[:-1]))
    
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
            raise StaticValueError(f"binary operation not supported: {node.op.__class__.__name__}")
    
    def visit_Ellipsis(self, _: Any) -> Type:
        return Type('...') 
    
    def visit_Constant(self, node: Union[ast.Constant, ast.Str, ast.NameConstant]) -> Type:
        if node.value is None:
            return Type('None')
        elif isinstance(node.value, type(...)):
            return self.visit_Ellipsis(None)
        if self.in_literal:
            return Type(repr(node.s if isinstance(node, ast.Str) else node.value))
        else:
            try:
                # unstring annotations as strings
                expr = _AnnotationStringParser().visit(node)
                if expr is node:
                    raise StaticValueError(f'unexpected {type(node.value).__name__}')
            except SyntaxError as e:
                raise StaticValueError('error in annotation') from e
            return self.visit(expr)
    
    visit_Str = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_List(self, node: ast.List) -> Type:
        # ast.List is used in Callable, but we do not fully support it at the moment.
        # TODO: are the lists supposed to only allowed in callables?
        return Type('', args=tuple(self.visit(el) for el in node.elts))

class _TypeInference(_EvalBaseVisitor['Type|None']):
    """
    Find the L{Type} of an expression.
    """

    #  MIT License

    #  2022 Gram

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included
    # in all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    _state: State
 
    def get_type(self, expr:ast.AST, path:list[ast.AST]) -> Type|None:
        try:
            return self.visit(expr, path)
        except StaticException as e:
            self._state.msg(f'type inference failed: {e.msg()}', ctx=e.node)
        except Exception as e:
            self._state.msg(f'unexpected {type(e).__name__} in type inference: {e}', ctx=expr)
            if __debug__:
                raise
        return None
        
    # AST expressions

    def visit_Constant(self, node: ast.Constant, path:list[ast.AST]) -> Type:
        if node.value is None:
            return Type('None')
        return Type(type(node.value).__name__)
    
    def visit_JoinedStr(self, node: ast.JoinedStr, path:list[ast.AST]) -> Type:
        return Type('str')
    
    # container types

    def visit_List(self, node: ast.List|ast.Set, path:list[ast.AST]) -> Type:
        clsname = type(node).__name__.lower()
        subtype = Type.Any
        for element_node in node.elts:
            element_type = self.get_type(element_node, path)
            if element_type is None:
                return Type(clsname)
            subtype = subtype.merge(element_type)
        if subtype.unknown:
            return Type(clsname)
        return Type(clsname, args=(subtype,))
    
    visit_Set = visit_List

    def visit_Tuple(self, node: ast.Tuple, path:list[ast.AST]) -> Type:
        subtypes: tuple[Type,...] = ()
        for element_node in node.elts:
            element_type = self.get_type(element_node, path)
            if element_type is None:
                return Type('tuple')
            subtypes += (element_type,)
        if not subtypes:
            return Type('tuple')
        return Type('tuple', args=subtypes)

    def visit_Dict(self, node: ast.Dict, path:list[ast.AST]) -> Type:
        keys_type = Type.Any
        unpack_indexes = set() 
        for i, key_node in enumerate(node.keys):
            if key_node is None:
                unpack_indexes.add(i)
                continue
            key_type = self.get_type(key_node, path)
            if key_type is None:
                key_type = Type.Any
                break
            keys_type = keys_type.merge(key_type)

        values_type = Type.Any
        for i, value_node in enumerate(node.values):
            if i in unpack_indexes:
                # TODO: we could do better here, it ignore unpacking for now.
                continue
            value_type = self.get_type(value_node, path)
            if value_type is None:
                value_type = Type.Any
                break
            values_type = values_type.merge(value_type)

        if keys_type.unknown and values_type.unknown:
            return Type('dict')
        if keys_type.unknown:
            keys_type = Type.Any
        if values_type.unknown:
            values_type = Type.Any
        return Type('dict', args=(keys_type, values_type))
    
    def visit_UnaryOp(self, node: ast.UnaryOp, path:list[ast.AST]) -> Type | None:
        if isinstance(node.op, ast.Not):
            return Type('bool')
        result = self.get_type(node.operand, path)
        if result is not None:
            # result = result.add_ass(Ass.NO_UNARY_OVERLOAD)
            return result
        return None

    def visit_BinOp(self, node: ast.BinOp, path:list[ast.AST]) -> Type | None:
        assert node.op
        lt = self.get_type(node.left, path)
        if lt is None:
            return None
        rt = self.get_type(node.right, path)
        if rt is None:
            return None
        if lt.name == rt.name == 'int':
            if isinstance(node.op, ast.Div):
                return Type('float')
            return lt
        if lt.name in ('float', 'int') and rt.name in ('float', 'int'):
            return Type('float')
        if lt.name == rt.name:
            return rt
        return None

    def visit_BoolOp(self, node: ast.BoolOp, path:list[ast.AST]) -> Type | None:
        assert node.op
        result = Type.Any
        for subnode in node.values:
            type = self.get_type(subnode, path)
            if type is None:
                return None
            result = result.merge(type)
        return result

    def visit_Compare(self, node: ast.Compare, path:list[ast.AST]) -> Type | None:
        if isinstance(node.ops[0], ast.Is):
            return Type('bool')
        # TODO: Use typeshed here to get precise type.
        return Type('bool')#, ass={Ass.NO_COMP_OVERLOAD})

    def visit_ListComp(self, node: ast.ListComp, path:list[ast.AST]) -> Type | None:
        return Type('list')

    def visit_SetComp(self, node: ast.SetComp, path:list[ast.AST]) -> Type | None:
        return Type('set')

    def visit_DictComp(self, node: ast.DictComp, path:list[ast.AST]) -> Type | None:
        return Type('dict')

    def visit_GeneratorExp(self, node: ast.GeneratorExp, path:list[ast.AST]) -> Type | None:
        return Type('Iterator', 'typing')
    
    def visit_Name_Store(self, node:ast.Name, path:list[ast.AST]) -> Type|None:
        ...

    def visit_Attribute_Store(self, node:ast.Attribute, path:list[ast.AST]) -> Type|None:
        ...
    
    def visit_Name_Load(self, node:ast.Name, path:list[ast.AST]) -> Type|None:
        try:
            defnode = self._state.goto_definition(node,
                        raise_on_ambiguity=self._raise_on_ambiguity,
                        follow_aliases=True)
        except StaticException as e: 
            self._state.msg(f'cannot infer definition of name {node.id!r}: {str(e)}', ctx=node)
            return None
        return self.get_type(defnode.node, path)

    def visit_Attribute_Load(self, node:ast.Attribute, path:list[ast.AST]) -> Type|None:
        valuetype = self.get_type(node.value, path)
        if valuetype is None:
            return None
        
        attrdefs: Sequence[Def]
        
        if valuetype.is_module and len(valuetype.args)==1:
            modname = valuetype.args[0].name
            modnode = self._state.get_module(modname)
            if modnode is None:
                self._state.msg(f'cannot infer attribute of unknown module {modname!r}', ctx=node)
                return None
            attrdefs = self._state.get_attribute(modnode, node.attr)
        else:
            include_ivars = True
            if valuetype.is_type and len(valuetype.args)==1:
                clslocation = valuetype.args[0].location
                clsqualname = (valuetype.args[0].qualname if 
                               valuetype.args[0].scope else 
                               f'builtins.{valuetype.args[0].name}')
                include_ivars = False
            else:
                # TODO:handle unions.
                # assume it's an instance of a type in the project
                clslocation = valuetype.location
                clsqualname = (valuetype.qualname if 
                               valuetype.scope else 
                               f'builtins.{valuetype.name}')
            
            if not clsqualname.startswith('builtins.') and str(clslocation) == '?':
                # nope it's not
                self._state.msg(f'cannot infer attribute of unknown type {clsqualname!r}', ctx=node)
                return None
            try:
                clsdefs = self._state.get_defs_from_qualname(clsqualname)
            except StaticException as e:
                self._state.msg(f'cannot infer attribute of type {clsqualname!r} {clslocation}: {e.msg()}', ctx=node)
                return None
            
            if len(clsdefs)>1:
                try:
                    # try to find the class with the same location as the Typr
                    clsnode = next(filter(
                        lambda d: isinstance(d, Cls) and 
                            clslocation == self._state.get_location(d),
                        self._state.get_defs_from_qualname(clsqualname)))
                except StopIteration:
                    # should report warning here,
                    # fallback to another definiton that doesn't match the location.
                    # for builtins for instance.
                    clsnode = clsdefs[-1]
            else:
                clsnode, = clsdefs            
            attrdefs = self._state.get_attribute(clsnode, 
                            node.attr, include_ivars=include_ivars)
        
        if len(attrdefs) > 1 and self._raise_on_ambiguity:
            raise StaticAmbiguity(
                node, f"{len(attrdefs)} potential definitions found",
                filename=self._state.get_filename(node)
            )
        return self.get_type(attrdefs[-1].node, path)


    def visit_Call(self, node:ast.Call, path:list[ast.AST]) -> Type|None:
        assert node.func
        functype = self.get_type(node.func, path)
        if functype:
            if functype.is_type and len(functype.args)==1:
                return functype.args[0]
            if functype.is_callable and len(functype.args)==2:
                # TODO: Find and match overloads
                return functype.args[1]
            self._state.msg(f'cannot infer call result of type: {functype.annotation}', ctx=node)
        return None

    # def visit_Subscript(self, node:ast.Subscript, path:list[ast.AST]) -> Type|None:
    #     ...

    # statements

    def visit_FunctionDef(self, node: ast.FunctionDef|ast.AsyncFunctionDef, path:list[ast.AST]) -> Type:
        # TODO: get better at callables
        argstype = Type.Any
        if node.returns is not None:
            returntype = _AnnotationToType(self._state, 
                            self._state.get_enclosing_scope(node)).visit(node.returns)
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: better support for async functions
            returntype = Type('Awaitable', 'typing', args=(Type.Any,))
        else:
            # TODO: Look at returns and infer that.
            returntype = Type.Any
        return Type.Callable.add_args(args=(argstype, returntype))._replace(
                    location=self._state.get_location(node))
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_Module(self, node: ast.Module, path:list[ast.AST]) -> Type:
        modname = self._state.get_qualname(node)
        return Type.Module.add_args(args=(
                    Type(modname, location=self._state.get_location(node)),
                ))
    
    def visit_ClassDef(self, node: ast.ClassDef, path:list[ast.AST]) -> Type:
        return Type.Type.add_args(
            args=(Type(node.name, self._state.get_qualname(
                            self._state.get_enclosing_scope(node)), 
                        location=self._state.get_location(node)),))