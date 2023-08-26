from __future__ import annotations

import ast
import sys
from typing import (
    Any,
    Iterator,
    Tuple,
    Union,
    List,
    TYPE_CHECKING,
    cast,
)
from inspect import Parameter

from .._lib.model import Type, Scope, Def, Mod, Func, Cls, Arg, LazySeq, NameDef, Var
from .._lib.shared import node2dottedname, ast_node_name
from .._lib.assignment import get_stored_value
from .._lib.exceptions import (
    NodeLocation,
    StaticException,
    StaticNameError,
    StaticValueError,
    StaticCodeUnsupported,
)
from .asteval import _EvalBaseVisitor

from beniget.beniget import BuiltinsSrc  # type: ignore

if TYPE_CHECKING:
    from .state import State

class _AnnotationStringParser(ast.NodeTransformer):
    """When given an expression, the node returned by L{ast.NodeVisitor.visit()}
    will also be an expression.
    If any string literal contained in the original expression is either
    invalid Python or not a singular expression, L{SyntaxError} is raised.
    """

    def __init__(self, filename:str|None) -> None:
        self.filename = filename

    def _parse_string(self, value: str, ctx:ast.AST) -> ast.expr:
        statements = ast.parse(value).body
        if len(statements) != 1:
            raise StaticValueError(ctx, "expected expression, found multiple statements", 
                                   filename=self.filename)
        (stmt,) = statements
        if isinstance(stmt, ast.Expr):
            # Expression wrapped in an Expr statement.
            expr = self.visit(stmt.value)
            assert isinstance(expr, ast.expr), expr
            return expr
        else:
            raise StaticValueError(ctx, "expected expression, found statement", 
                                   filename=self.filename)

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        value = self.visit(node.value)
        if isinstance(value, ast.Name) and value.id == "Literal":
            # Literal[...] expression; don't unstring the arguments.
            slice = node.slice
        elif isinstance(value, ast.Attribute) and value.attr == "Literal":
            # typing.Literal[...] expression; don't unstring the arguments.
            slice = node.slice
        else:
            # Other subscript; unstring the slice.
            slice = self.visit(node.slice)
        return ast.fix_missing_locations(
            ast.copy_location(ast.Subscript(value, slice, node.ctx), node))

    # For Python >= 3.8:

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        value = node.value
        if isinstance(value, str):
            return ast.fix_missing_locations(
                ast.copy_location(self._parse_string(value, node), node))
        else:
            const = self.generic_visit(node)
            assert isinstance(const, ast.Constant), const
            return const

    # For Python < 3.8:

    def visit_Str(self, node: ast.Str) -> ast.expr:
        return ast.fix_missing_locations(
            ast.copy_location(self._parse_string(node.s, node), node))


def _union(*args: Union[Type, str]) -> Type:
    new_args: tuple[Type, ...] = ()
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

    _redirects = {
        'typing.Dict'           :   'builtins.dict', 
        'typing.Tuple'          :   'builtins.tuple',
        'typing.List'           :   'builtins.list',
        'typing.Set'            :   'builtins.set',
        'typing.FrozenSet'      :   'builtins.frozenset',
        'typing.Text'           :   'builtins.str',
        'typing.DefaultDict'    :   'collections.defaultdict',
        'typing.NamedTuple'     :   'collections.namedtuple', 
    }

    def __init__(self, state: State, scope: Scope) -> None:
        self.state = state
        self.scope = scope
        self.in_literal = False

    def generic_visit(self, node: ast.AST) -> Any:
        raise StaticValueError(node, f"unexcepted node in annotation: {node}", 
                               filename=self.state.get_filename(node))

    def visit(self, expr: ast.AST) -> Type:
        """
        Callers should catch any L{Exception}.
        """
        return super().visit(expr)

    def visit_Name(self, node: ast.Name) -> Type:
        qualname = self.state.expand_expr(node) or self.state.expand_name(
            self.scope, node.id
        )
        if qualname:
            qualname = self._redirects.get(qualname, qualname)
            module, _, name = qualname.rpartition(".")
            return Type(name, module).add_meta(location=self.state.get_location(node))
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        elif node.id in BuiltinsSrc:
            # the builtin module might not be in the system
            return Type(node.id).add_meta(location=self.state.get_location(node))
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
            raise StaticValueError(
                node,
                desrc="illegal expression in annotation",
                filename=self.state.get_filename(node),
            )

        qualname = self.state.expand_expr(node) or self.state.expand_name(
            self.scope, ".".join(dottedname)
        )
        if qualname:
            qualname = self._redirects.get(qualname, qualname)
            module, _, name = qualname.rpartition(".")
            return Type(name, module)
            # TODO: This ignores the fact that the parent of the imported symbol migt be a class.
        else:
            # TODO: Leave a warning, the name is unbound
            raise StaticNameError(node, filename=self.state.get_filename(node))

    def visit_Subscript(self, node: ast.Subscript) -> Type:
        left = self.visit(node.value)
        if left.is_literal:
            self.in_literal = True
        try:
            if sys.version_info < (3,9):
                if isinstance(node.slice, ast.Index):
                    slicevalue = node.slice.value
                else:
                    # raises
                    self.generic_visit(node.slice) 
            else:
                slicevalue = node.slice
            if isinstance(slicevalue, ast.Tuple):
                args = [self.visit(el) for el in slicevalue.elts]
                left = left._replace(args=args)
            else:
                arg = self.visit(slicevalue)
                if arg:
                    left = left._replace(args=[arg])
            # nested literal are considered invalid annotations
        except StaticException as e:
            self.state.msg(e.msg(), ctx=e.node)
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
            raise StaticValueError(node, 
                f"binary operation not supported: {node.op.__class__.__name__}",
                filename=self.state.get_filename(node)
            )

    def visit_Ellipsis(self, _: Any) -> Type:
        return Type("...")

    def visit_Constant(
        self, node: Union[ast.Constant, ast.Str, ast.NameConstant, ast.Bytes, ast.Num]
    ) -> Type:
        if isinstance(node, (ast.Str, ast.Bytes)):
            value: object = node.s
        elif isinstance(node, ast.Num):
            value = node.n
        else:
            value = node.value
        if value is None:
            return Type("None")
        elif isinstance(value, type(...)):
            return self.visit_Ellipsis(None)
        if self.in_literal:
            return Type(repr(value))
        else:
            try:
                # unstring annotations as strings
                expr = _AnnotationStringParser(self.state.get_filename(node)).visit(node)
                if expr is node:
                    raise StaticValueError(node, f"unexpected {type(node.value).__name__}",
                            filename=self.state.get_filename(node))
            except SyntaxError as e:
                raise StaticValueError(node, "error in annotation", 
                        filename=self.state.get_filename(node)) from e
            return self.visit(expr)

    visit_Str = visit_Constant
    visit_Bytes = visit_Constant
    visit_Num = visit_Constant
    visit_NameConstant = visit_Constant


    def visit_List(self, node: ast.List) -> Type:
        # ast.List is used in Callable, but we do not fully support it at the moment.
        # TODO: are the lists supposed to only allowed in callables?
        return Type("", args=tuple(self.visit(el) for el in node.elts))


class _TypeInference(_EvalBaseVisitor["Type|None"]):
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

    def get_type(self, expr: ast.AST, path: list[ast.AST]) -> Type | None:
        try:
            return self.visit(expr, path)
        except StaticException as e:
            self._state.msg(f"type inference failed: {e.msg()}", ctx=expr)
        except Exception as e:
            self._state.msg(
                f"unexpected {type(e).__name__} in type inference: {e}", ctx=expr
            )
            if __debug__:
                raise
        return None

    #########################################
    ###      expressions                  ###
    #########################################

    def visit_Constant(
        self, node: Union[ast.Constant, ast.Str, ast.NameConstant, ast.Bytes, ast.Num]
    ) -> Type:
        if isinstance(node, (ast.Str, ast.Bytes)):
            value: object = node.s
        elif isinstance(node, ast.Num):
            value = node.n
        else:
            value = node.value
        if value is None:
            return Type("None")
        return Type(type(value).__name__)
    
    visit_Str = visit_Constant
    visit_Bytes = visit_Constant
    visit_Num = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_JoinedStr(self, node: ast.JoinedStr, path: list[ast.AST]) -> Type:
        return Type("str")

    def visit_List(self, node: ast.List | ast.Set, path: list[ast.AST]) -> Type:
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

    def visit_Tuple(self, node: ast.Tuple, path: list[ast.AST]) -> Type:
        subtypes: tuple[Type, ...] = ()
        for element_node in node.elts:
            element_type = self.get_type(element_node, path)
            if element_type is None:
                return Type("tuple")
            subtypes += (element_type,)
        if not subtypes:
            return Type("tuple")
        return Type("tuple", args=subtypes)

    def visit_Dict(self, node: ast.Dict, path: list[ast.AST]) -> Type:
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
            return Type("dict")
        if keys_type.unknown:
            keys_type = Type.Any
        if values_type.unknown:
            values_type = Type.Any
        return Type("dict", args=(keys_type, values_type))

    def visit_UnaryOp(self, node: ast.UnaryOp, path: list[ast.AST]) -> Type | None:
        if isinstance(node.op, ast.Not):
            return Type("bool")
        result = self.get_type(node.operand, path)
        if result is not None:
            # result = result.add_ass(Ass.NO_UNARY_OVERLOAD)
            return result
        return None

    def visit_BinOp(self, node: ast.BinOp, path: list[ast.AST]) -> Type | None:
        assert node.op
        lt = self.get_type(node.left, path)
        if lt is None:
            return None
        rt = self.get_type(node.right, path)
        if rt is None:
            return None
        if lt.qualname == rt.qualname == "builtins.int":
            if isinstance(node.op, ast.Div):
                return Type("float")
            return lt
        if lt.qualname in ("builtins.float", "builtins.int") and \
            rt.qualname in ("builtins.float", "builtins.int"):
            return Type("float")
        if lt.qualname == rt.qualname:
            return rt
        return None

    def visit_BoolOp(self, node: ast.BoolOp, path: list[ast.AST]) -> Type | None:
        assert node.op
        result = Type.Any
        for subnode in node.values:
            type = self.get_type(subnode, path)
            if type is None:
                return None
            result = result.merge(type)
        return result

    def visit_Compare(self, node: ast.Compare, path: list[ast.AST]) -> Type | None:
        if isinstance(node.ops[0], ast.Is):
            return Type("bool")
        # TODO: Use typeshed here to get precise type.
        return Type("bool")  # , ass={Ass.NO_COMP_OVERLOAD})

    def visit_ListComp(self, node: ast.ListComp, path: list[ast.AST]) -> Type | None:
        return Type("list")

    def visit_SetComp(self, node: ast.SetComp, path: list[ast.AST]) -> Type | None:
        return Type("set")

    def visit_DictComp(self, node: ast.DictComp, path: list[ast.AST]) -> Type | None:
        return Type("dict")

    def visit_GeneratorExp(
        self, node: ast.GeneratorExp, path: list[ast.AST]
    ) -> Type | None:
        return Type("Iterator", "typing")

    def visit_Call(self, node: ast.Call, path: list[ast.AST]) -> Type | None:
        assert node.func
        functype = self.get_type(node.func, path)
        if functype:
            if functype.is_type and len(functype.args) == 1:
                return functype.args[0]
            if functype.is_callable and len(functype.args) == 2:
                # TODO: Find and match overloads
                returntype = functype.args[1]
                if returntype.unknown:
                    return None
                return returntype
            raise StaticValueError(
                node,
                f"cannot infer call result of type: {functype!r}",
                filename=self._state.get_filename(node),
            )
        return None

    def visit_Subscript(self, node: ast.Subscript, path: list[ast.AST]) -> Type | None:
        assert node.value
        valuetype = self.get_type(node.value, path)
        if valuetype is None:
            return None
        if valuetype.qualname in ("builtins.str", "builtins.bytes"):
            return valuetype
        if valuetype.qualname in ("builtins.dict",) and len(valuetype.args) == 2:
            return valuetype.args[1]
        if valuetype.qualname in ("builtins.list",) and len(valuetype.args) == 1:
            return valuetype.args[0]
        if valuetype.qualname in ("builtins.tuple",):
            if len(valuetype.args) == 0:
                return None
            if len(valuetype.args) == 2 and valuetype.args[1].annotation == "...":
                return valuetype.args[0]
            try:
                indexvalue = self._state.literal_eval(node.slice)
            except StaticException:
                pass
            else:
                if not isinstance(indexvalue, int):
                    return None
                try:
                    return valuetype.args[indexvalue]
                except IndexError:
                    return None

            newtype = Type.Any
            for t in valuetype.args:
                newtype = newtype.merge(t)
            return newtype
        return None
    
    #########################################
    ###      jumps                        ###
    #########################################

    def visit_Name_Store(
        self, node: ast.Name | ast.Attribute, path: list[ast.AST]
    ) -> Type | None:
        # doesn't support augmented assignments
        try:
            assign = self._state.get_parent_instance(node, (ast.Assign, ast.AnnAssign))
        except StaticException as e:
            raise StaticCodeUnsupported(
                node, "name", filename=self._state.get_filename(node)
            ) from e
        if isinstance(assign, ast.AnnAssign):
            ann = self._replace_typevars_by_any(_AnnotationToType(
                self._state, self._state.get_enclosing_scope(node)
            ).visit(assign.annotation))
            if not ann.unknown:
                return ann
        value = get_stored_value(node, assign=assign)  # type:ignore[arg-type]
        if value is not None:
            # TODO: Detect if the given assignment is an implicit
            # type alias and use _AnnotationToType in these cases.
            return self.get_type(value, path)
        raise StaticValueError(
            node,
            # it must be bogus because get_stored_value() already raises on unsupported constructs.
            f'no known type for {"name" if isinstance(node, ast.Name) else "attribute"}',
            filename=self._state.get_filename(node),
        )

    visit_Attribute_Store = visit_Name_Store

    def visit_Name_Load(
        self, node: ast.Name | ast.alias, path: list[ast.AST]
    ) -> Type | None:
        try:
            name_defs = self._state.goto_defs(node)
        except StaticException as e:
            raise StaticValueError(
                node,
                f"cannot find definition of name {ast_node_name(node)!r}: {str(e)}",
                filename=self._state.get_filename(node),
            ) from e

        newtype = Type.Any
        for d in name_defs:
            other = self.get_type(d.node, path)
            if other:
                newtype = newtype.merge(other)
        if newtype.unknown:
            if len(name_defs)>1:
                raise StaticValueError(
                    node,
                    f"found {len(name_defs)} definition for name {ast_node_name(node)!r}, but none of them have a known type",
                    filename=self._state.get_filename(node),
                )
            else:
                raise StaticValueError(
                    node,
                    f"found a definition for name {ast_node_name(node)!r}, but it does not have a known type",
                    filename=self._state.get_filename(node),
                )
        return newtype

    visit_alias = visit_Name_Load

    def visit_arg(self, node: ast.arg, path: list[ast.AST]) -> Type | None:
        arg_def: Arg = self._state.get_def(node)  # type:ignore
        if arg_def.node.annotation is not None:
            try:
                annotation = self._replace_typevars_by_any(_AnnotationToType(
                    self._state, LazySeq(self._state.get_all_enclosing_scopes(node))[1]
                ).visit(arg_def.node.annotation))
            except StaticException:
                pass
            else:
                if not annotation.unknown:
                    if arg_def.kind == Parameter.VAR_POSITIONAL:
                        annotation = Type("tuple", args=[annotation, Type("...")])
                    if arg_def.kind == Parameter.VAR_KEYWORD:
                        annotation = Type("dict", args=[Type("str"), annotation])
                    return annotation
        if (
            arg_def.default is not None
            and getattr(arg_def.default, "value", object()) is not None
        ):
            return self.get_type(arg_def.default, path)
        if arg_def.kind == Parameter.VAR_POSITIONAL:
            return Type("tuple")
        if arg_def.kind == Parameter.VAR_KEYWORD:
            return Type("dict", args=[Type("str"), Type.Any])
        return None
    
    def visit_Attribute_Load(
        self, node: ast.Attribute, path: list[ast.AST]
    ) -> Type | None:
        valuetype = self.get_type(node.value, path)
        if valuetype is None:
            return None
        return self._get_type_attribute(valuetype, node.attr, node, path)

    #########################################
    ###      statements                   ###
    #########################################

    def visit_FunctionDef(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, path: list[ast.AST]
    ) -> Type:
        # TODO: get better at callables
        argstype = Type.Any
        if node.returns is not None:
            returntype = self._replace_typevars_by_any(_AnnotationToType(
                self._state, self._state.get_enclosing_scope(node)
            ).visit(node.returns))
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: better support for async functions
            returntype = Type("Awaitable", "typing", args=(Type.Any,))
        else:
            returntype = Type.Any
        if any(
            self._state.expand_expr(n) in ("property", "builtins.property")
            for n in node.decorator_list
        ):
            return returntype
        return Type.Callable.add_args(args=(argstype, returntype)).add_meta(
            qualname=self._state.get_qualname(node),
            location=self._state.get_location(node), 
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Module(self, node: ast.Module, path: list[ast.AST]) -> Type:
        modname = self._state.get_qualname(node)
        return Type.ModuleType.add_meta(
            qualname=modname, 
            location=self._state.get_location(node))

    def visit_ClassDef(self, node: ast.ClassDef, path: list[ast.AST]) -> Type:
        return Type.TypeType.add_args(
            args=(Type(
                    node.name,
                    self._state.get_qualname(self._state.get_enclosing_scope(node)),
                    location=self._state.get_location(node),),)
        )

    #########################################
    ###      type inference helpers       ###
    #########################################

    def _get_type_attribute(
        self, valuetype: Type, 
        attr: str, 
        ctx: ast.AST, 
        path: list[ast.AST]
    ) -> Type | None:
        """
        Get the type of an attribute access ``attr`` on the given ``valuetype``.
        """
        scopedefs: List[Def] = []
        attrdefs: List[Def] = []
        for type, definition in self._flatten_typedefs(
            valuetype, set()
        ):
            if not isinstance(definition, (Mod, Cls)):
                continue

            scopedefs.append(definition)
            try:
                defs = self._state.get_attribute( # type: ignore
                    definition, attr, include_ivars=not type.is_type # type: ignore
                )
            except StaticException:
                continue

            attrdefs.extend(defs)

        if len(attrdefs) == 0:
            if len(scopedefs)>1:
                raise StaticValueError(
                    ctx,
                    f"attribute {attr} not found "
                    f'in any of {[f"{d.name()}:{self._state.get_location(d)}" for d in scopedefs]}',
                    filename=self._state.get_filename(ctx),
                )
            elif len(scopedefs)==1:
                raise StaticValueError(
                    ctx,
                    f"attribute {attr} not found "
                    f'in {next(f"{d.name()}:{self._state.get_location(d)}" for d in scopedefs)}',
                    filename=self._state.get_filename(ctx),
            )
            else:
                raise StaticValueError(
                    ctx,
                    f"no valid attribute access namespace found on type {valuetype.annotation}, "
                    f"can't look for attribute {attr}",
                    filename=self._state.get_filename(ctx),)

        newtype = Type.Any
        for definition in attrdefs:
            othertype = self.get_type(definition.node, path)
            if othertype is not None:
                newtype = newtype.merge(othertype)
        
        if newtype.unknown:
            if len(attrdefs) > 1:
                raise StaticValueError(
                    ctx,
                    f"found {len(attrdefs)} definitions for attribute {attr!r}, but none of them have a known type",
                    filename=self._state.get_filename(ctx),
                )
            else:
                raise StaticValueError(
                    ctx,
                    f"found a definition for attribute {attr!r}, but it does not have a known type",
                    filename=self._state.get_filename(ctx),
                )
        return newtype
    
    def _replace_typevars_by_any(self, type: Type) -> Type:
        """
        Sine we don't support typevars at the moment, we simply replace them by Any :/
        """
        # lectures about unification of typevars: 
        # https://stackoverflow.com/questions/65362422/type-unification-algorithm-in-python-how-to-reject-unifya-b-int-int
        # https://gist.github.com/dhilst/b5b198af93302ade61ccbfe3b094621a
        # https://eli.thegreenplace.net/2018/unification/
        # https://github.com/caterinaurban/Typpete/blob/master/typpete/src/annotation_resolver.py
        # https://github.com/pfalcon/picompile
        # https://github.com/eliphatfs/typhon/tree/master/typhon/core/type_system/intrinsics
        # https://github.com/serge-sans-paille/tog/blob/master/tog.py#L391

        try:
            definition = self._get_typedef(type) # this can raise StaticException
            if isinstance(definition, Var):
                assign = self._state.get_parent(definition)
                if isinstance(assign, ast.Assign):
                    if isinstance(assign.value, ast.Call):
                        name = node2dottedname(assign.value.func)
                        if name:
                            if self._state.expand_name(self._state.get_root(definition), 
                                                    '.'.join(name)) == 'typing.TypeVar':
                                return Type.Any
        except StaticException as e:
            pass
        
        subtypes = [self._replace_typevars_by_any(t) for t in type.args]
        if all(s.unknown for s in subtypes):
            subtypes = []
        return type._replace(args=subtypes)
    
    def _get_typedef(self, typ:Type) -> Def:
        """
        Find the definition of a Type.
        """
        # supports classes, modules, functions and variables at the moment.
        location:NodeLocation|None
        qualname:str
        if typ.is_module:
            qualname = typ.get_meta('qualname', str) # type:ignore[assignment]
            if qualname is None:
                raise StaticValueError(typ, "no module definition")
            hint:type[Def]|tuple[type[Def],...] = Mod
            location = typ.get_meta('location', NodeLocation)
        elif typ.is_callable:
            qualname = typ.get_meta('qualname', str) # type:ignore[assignment]
            if qualname is None:
                raise StaticValueError(typ, "no function definition")
            hint = Func
            location = typ.get_meta('location', NodeLocation)
        else:
            location = typ.location
            qualname = typ.qualname
            if '.' not in qualname:
                raise StaticValueError(typ, f"won't find anything for type {typ}")
            hint = (Cls, Var)
        if hint is Mod:
            m = self._state.get_module(qualname)
            if m is None:
                raise StaticValueError(typ, 
                    f"unknown module {typ.name!r}",)
            return m
        else:
            return self._find_typedef(qualname, hint=hint, location=location)


    def _find_typedef(self, qualname: str, *, 
                      hint:type[Def]|tuple[type[Def],...], 
                      location: NodeLocation|None=None) -> Def:
        """
        Get the definition of the object qualified by the given name, 
        type hint and location or raise an exception.
        """
        try:
            defs = self._state.get_defs_from_qualname(qualname)
        except StaticException as e:
            raise StaticValueError(
                e.node, f"unknown symbol: {qualname!r}: {e.msg()}", filename=e.filename
            ) from e
        init_defs = defs
        defs = list(filter(lambda d: isinstance(d, hint), defs))
        if len(defs) == 0:
            raise StaticValueError(
             f"cannot find symbol {qualname!r} of type {hint} but found {len(init_defs)} other definitions",
            )
        if len(defs) > 1 and location is not None:
            try:
                # try to find the class with the same location as the Type
                # TODO: It might be a type alias?
                defs = [next(filter(lambda d: location == self._state.get_location(d), defs,)) ]
            except StopIteration:
                # should report warning here,
                # fallback to another definiton that doesn't match the location.
                # for builtins for instance.
                pass
        *_, node = defs
        return node

    def _flatten_typedefs(
        self, valuetype: Type, seen: set[Type]
    ) -> Iterator[Tuple[Type, Def]]:
        """
        Get the definition of each resolvable 
        top-level types in this type instance.
        """
        if valuetype in seen:
            return
        seen.add(valuetype)

        if valuetype.is_union:
            for subtype in valuetype.args:
                nested = self._flatten_typedefs(
                    subtype, seen.copy()
                )
                while 1:
                    try:
                        yield next(nested)
                    except StopIteration:
                        break
                    except StaticException as e:
                        self._state.msg(f'incomplete union: {e.msg()}', ctx=e.location())

        elif valuetype.is_literal:
            try:
                val = ast.literal_eval(valuetype.args[0].name)
            except Exception:
                return
            if val is None:
                yield valuetype, self._find_typedef("types.NoneType", hint=Cls)
            else:
                yield valuetype, self._find_typedef(f"builtins.{type(val).__name__}", hint=Cls)
        elif valuetype.is_type:
            if len(valuetype.args) == 1:
                # Class variable attribute
                yield valuetype, self._get_typedef(valuetype.args[0])
            else:
                self._state.msg(f'incomplete type: {valuetype.annotation}', ctx=valuetype.location)
        else:
            # assume it's an instance of a type in the project
            yield valuetype, self._get_typedef(valuetype)
