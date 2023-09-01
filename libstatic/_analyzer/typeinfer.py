from __future__ import annotations

import ast
from collections import deque
from functools import cache, cached_property
import inspect
from itertools import chain
import itertools
import sys
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    Tuple,
    Union,
    List,
    TYPE_CHECKING,
    cast,
)
from inspect import Parameter

from .._lib.model import (Scope, Def, Mod, 
                          Func, Cls, Arg, LazySeq, 
                          FrozenDict, ChainMap, LazyMap, 
                          NameDef, Var)
from .._lib.ivars import is_instance_method
from .._lib.shared import node2dottedname, ast_node_name
from .._lib.assignment import get_stored_value
from .._lib.exceptions import (
    NodeLocation,
    StaticException,
    StaticNameError,
    StaticValueError,
    StaticCodeUnsupported,
    StaticStateIncomplete,
)
from .asteval import _EvalBaseVisitor

import attr as attrs
from attrs import validators

from beniget.beniget import BuiltinsSrc, ordered_set  # type: ignore

if TYPE_CHECKING:
    from .state import State
    from typing import NoReturn

def _raise(e:Exception) -> NoReturn: 
    raise e

_T = TypeVar('_T')

__docformat__ = 'google'

@cache
def find_typedef(state:State, qualname: str, *, 
                 hint:type[Def]|tuple[type[Def],...]=Def, 
                 location: NodeLocation|None=None) -> NameDef:
    """
    Get the definition of the object qualified by the given name, 
    type hint and location or raise an exception.

    Wrapper arround `get_defs_from_qualname`.
    """
    try:
        defs = state.get_defs_from_qualname(qualname)
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
            # try to find the definition with the given location
            defs = [next(filter(lambda d: location == state.get_location(d), defs,)) ]
        except StopIteration:
            # should report warning here,
            # fallback to another definiton that doesn't match the location.
            pass
    *_, node = defs
    return node

# This class has beeen adapted from the 'astypes' project.
@attrs.s(frozen=True, auto_attribs=True)
class Type:
    """
    Internal implementation of `libstatic.Type`.
    """
    
    name: str = attrs.ib(validator=[validators.instance_of(str),
                                    validators.min_len(1)]) # type: ignore
    scope: str = ''
    """The scope where the type is defined. This is often a module, 
    but it migth be a class or a function in some cases.

    For example, `typing` if the type is `Iterable`.
    Empty string for built-ins or other special cases.
    """

    args: Sequence['Type'] = attrs.ib(factory=tuple, kw_only=True)
    
    if not TYPE_CHECKING:
        # mypy is not very smart with the converter option :/
        args: Sequence['Type'] = attrs.ib(factory=tuple, converter=tuple, kw_only=True)

    location: NodeLocation = attrs.ib(factory=NodeLocation, kw_only=True, eq=False, repr=False)
    """
    The location of the node that defined this type.
    It's set to an unknown location by default so special 
    types can be created dynamically. 
    """

    meta: Mapping[str, object] = attrs.ib(factory=FrozenDict, kw_only=True, eq=False)
    """
    Stores meta information when the type 
    annotations are not expressive enougth.
    Used for intermediate inference steps.
    """
    if not TYPE_CHECKING:
        meta: Mapping[str, object] = attrs.ib(factory=FrozenDict, converter=FrozenDict, kw_only=True, hash=False)
    
    definition: Def|None = attrs.ib(default=None, eq=False, hash=False)
    """
    The type symbol definition, if it has one. 
    Can be None for builtins (if the builtins module is not in the system) 
    and other special cases like Callable or Unions.
    """

    # Special types:
    Any: ClassVar[Type]
    Union: ClassVar[Type]
    TypeType: ClassVar[Type]
    Callable: ClassVar[Type]
    ModuleType: ClassVar[Type]
    Literal: ClassVar[Type]
    Optional: ClassVar[Type]
    overload: ClassVar[Type]

    @property
    def qualname(self) -> str:
        scope = self.scope
        if scope:
            return f"{scope}.{self.name}"
        elif self.name in BuiltinsSrc:
            return f"builtins.{self.name}"
        else:
            return self.name

    @property
    def unknown(self) -> bool:
        return (self.qualname == 'typing.Any' and self.get_meta('unknown', bool) is True)

    @property
    def is_union(self) -> bool:
        return self.qualname == 'typing.Union'
    
    @property
    def is_type(self) -> bool:
        return self.qualname == 'typing.Type'
    
    @property
    def is_optional(self) -> bool:
        return self.qualname == 'typing.Optional'
    
    @property
    def is_none(self) -> bool:
        return self.qualname == 'builtins.None'
    
    @property
    def is_callable(self) -> bool:
        return self.qualname == 'typing.Callable'
    
    @property
    def is_module(self) -> bool:
        return self.qualname == 'types.ModuleType'
    
    @property
    def is_typevar(self) -> bool:
        return self.name.startswith('@TypeVar')

    @property
    def is_protocol(self) -> bool:
        return self.get_meta('is_protocol', bool) or False

    @property
    def is_overload(self) -> bool:
        """
        >>> t = Type.overload.add_args(args=[Type.Callable.add_args([Type('int', 'builtins'), Type('int', 'builtins'),]), 
        ... Type.Callable.add_args([Type('str', 'builtins'), Type('str', 'builtins'),])])
        >>> print(t.annotation)
        (int) -> int | (str) -> str
        >>> assert t.is_overload
        """
        return self.name == '@overload'
    
    @property
    def is_literal(self) -> bool:
        """
        A literal type means it's literal values can be recovered with:
        
        >>> type = Type.Literal.add_args(args=[Type('"val"')])
        >>> ast.literal_eval(type.args[0].name)
        'val'
        """
        return self.qualname in ('typing.Literal', 
                                 'typing._extensions.Literal')

    def merge(self, other: 'Type') -> 'Type':
        """Get a union of the two given types.

        If any of the types is unknown, the other is returned.
        When possible, the type is simplified. For instance, ``int | int`` will be
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
    
    def _replace(self, **changes:str|Sequence[Type]|NodeLocation|dict) -> Type:
        return attrs.evolve(self, **changes) # type:ignore

    def add_args(self, args: Iterable[Type]) -> Type:
        """
        Get a copy of the Type with the given args added in the list of args.
        """
        return self._replace(args=tuple(chain(self.args, args)))

    def add_meta(self, **meta:object) -> Type:
        """
        Get a copy of the Type with the given meta informations updated.
        """
        return self._replace(meta={**self.meta, **meta})
    
    def get_meta(self, key:str, typ:type[_T]) -> _T|None:
        # """
        # Valid keys currently are 
        # - 'qualname' or 'location'. 
        #     Both are the respective qualname and location of the wrapped 
        #     function or module for `Callable` and `ModuleType`. 
        #     Set in the case of a known function or module only. 
        #     Won't be set for explicit 'Callable' annotations for instance, only for the once
        #     created from ast.FunctionDef instances, idem for modules.
        # - 'unknown'. 
        #     Set on generated Any types to differenciate an unknown type from an
        #     explicit typing.Any annotation, which is known to be Any, so should
        #     not be considered unknown.
        # """
        val = self.meta.get(key)
        if val is None:
            return None
        if not isinstance(val, typ):
            raise TypeError(f'expected {typ}, got {type(val)}')
        return val

    @property
    def supertype(self) -> Type | None:
        mro: Sequence[Type] | None = self.get_meta('mro', LazySeq)
        if mro: 
            return mro[0]
        return None

    def supertype_of(self, other: 'Type') -> bool:
        # TODO: use a mapping of type-promotion instead of this.
        if self.name == 'float' and other.name == 'int':
            return True
        
        # bottom type match
        if self.name in ('Any', 'object'):
            return True
        
        # union match
        if self.is_union:
            for arg in self.args:
                if arg.supertype_of(other):
                    return True
        
        if other.is_union:
            # all the other types must be a subtype of self.
            return all(self.supertype_of(o) for o in other.args)

        # exact match        
        if self.name == other.name and self.scope == other.scope and self.args == other.args:
            return True
        
        # Check protocol based matches.
        if self.is_protocol:
            members: Mapping[str, Type]|None = self.get_meta('members', LazyMap)
            other_members: Mapping[str, Type]|None = other.get_meta('members', LazyMap)
            if members is not None and other_members is not None:
                for name, membertype in members.items():
                    other_membertype = other_members.get(name)
                    if other_membertype is None:
                        # the name is not declared
                        break
                    if membertype.is_callable and other_membertype.qualname == 'builtins.None':
                        # the name is declared but assigned to None, 
                        # this means it does NOT implement that method.
                        break
                    # due to the incompleteness of our type inferrer, we cannot check more
                    # precise protocol members types. So delcaring a name is enougth to implement
                    # a protocol at this time.
                else:
                    return True

        # seek for superclasses
        supertype = other.supertype
        if supertype is not None:
            return self.supertype_of(supertype)
    
        return False
    
    @cached_property
    def annotation(self) -> str:
        """Represent the type as a string suitable for type annotations.

        The string is a valid Python 3.10 expression.
        For example, ``str | dict[str, Any]``.
        """
        if self.is_union or self.is_overload:
            return ' | '.join(arg.annotation for arg in self.args)
        name = self.name
        if self.args:
            if self.is_callable:
                def format_arg(p:Type) -> str:
                    ann = p.annotation
                    keyword = p.get_meta('keyword', str)
                    if keyword:
                        return f'{keyword}:{ann}'
                    else:
                        return ann
                *argstypes, rtype = self.args
                paramtypes = ', '.join(format_arg(arg) for arg in argstypes)
                return f'({paramtypes}) -> {rtype.annotation}'
            else:
                args = ', '.join(arg.annotation for arg in self.args)
                return f'{name}[{args}]'
        return name
    
    @cached_property
    def long_annotation(self) -> str:
        """
        Like `annotation` but returns the type with qualified names.
        """
        if self.is_union:
            return ' | '.join(arg.long_annotation for arg in self.args)
        # special case names that raises a SyntaxError
        name = self.qualname if self.qualname not in (
            'builtins.None', 'builtins.True', 'builtins.False') else self.name
        if self.args:
            if self.is_callable:
                *argstypes, rtype = self.args
                paramtypes = ', '.join(arg.long_annotation for arg in argstypes)
                args = f'[{paramtypes}], {rtype.long_annotation}'
            else:
                args = ', '.join(arg.long_annotation for arg in self.args)
            return f'{name}[{args}]'
        return name

def AnnotationType(state:State, qualname:str) -> Type:
    """
    Create a `Type` from it's qualified name.

    :raises StaticException: If the name is not in the system and the qualname
        does not refer to a builtin or a typing name. In these cases, a type with no
        definition is returned because we can still account for it
        even if the modules 'typing' or 'buitlins' are not in the system.
    """
    try:
        typedef = find_typedef(state, qualname)
    except StaticException:
        return SimpleType(qualname)
        # if qualname.startswith('builtins.') or \
        #     qualname.startswith('typing.') or qualname == 'types.ModuleType':
            
        # raise
    if isinstance(typedef, Cls):
        return ClsType(state, typedef)
    elif typedef is not None:
        # Else it could be a type alias/typevar etc.
        return SymbolType(state, typedef)

def SimpleType(qualname:str) -> Type:
    """
    Create a `Type` that is not in the system.
    """
    module, _, name = qualname.rpartition(".")
    return Type(name, module)

def SymbolType(state:State, definition:NameDef) -> Type:
    """
    Create a `Type` that is defined in the system but that's not a classdef.
    """
    name = definition.name()
    scopedef = state.get_enclosing_scope(definition)
    if scopedef is not None:
        scope = state.get_qualname(scopedef)
    else:
        # rare cases where a module is used in an annotation?
        scope = ''
    location = state.get_location(definition)
    return Type(name=name, 
                scope=scope, 
                definition=definition, 
                location=location)

def ClsType(state:State, definition:Cls) -> Type:
    """
    Create a `Type` from a classdef.
    """
    return SymbolType(state, definition).add_meta(
            is_protocol=any(state.expand_expr(n) in ('typing.Protocol', 
                                                            'typing_extensions.Protocol')
                            for n in definition.node.bases), 
            mro=SuperTypes(state, definition),
            members=MembersTypes(state, definition),
            )

def SuperTypes(state:State, definition:Cls) -> Sequence[Type]:
    """
    Fetch the super-types in mro order of this classdef.
    """
    def get_type(o:Cls|str) -> Type:
        if isinstance(o, Cls):
            return ClsType(state, o)
        else:
            # not in the system
            return SimpleType(o)

    return LazySeq((get_type(d) for d in 
                    state.get_mro(definition, include_unknown=True, include_self=False)))

def _merge_types(state:State, defs:Sequence[NameDef|None]) -> Type:
    nt = Type.Any
    for d in (d for d in defs if d):
        nt = nt.merge(state.get_type(d) or Type.Any) # type: ignore
    return nt

def unwrap_type_classdef(typ:Type) -> Def:
    # should onyl be called for typing.Type types.
    assert typ.is_type
    if len(typ.args) == 1:
        # Class variable attribute
        definition = typ.args[0].definition
        if definition is None:
            raise StaticNameError(typ.args[0].qualname, filename='<missing>')
        return definition
    elif typ.args:
       raise StaticValueError(typ.get_meta('location', NodeLocation), 
                              f'Type has to many argument: {typ.annotation}')
    else:
       raise StaticValueError(typ.get_meta('location', NodeLocation), 
                              f'Type has no argument: {typ.annotation}')

def MembersTypes(state:State, definition:Cls) -> Mapping[str, Type]:
    """
    Get a lazy mapping of all types of the members of this class, including inherited.
    """
    return LazyMap(((k, _merge_types(state, v)) for k,v in 
                        ChainMap([state.get_locals(definition), 
                                  state.get_ivars(definition)]).items()))

def BuiltinType(state:State, name:str) -> Type:
    return AnnotationType(state, f'builtins.{name}')

Type.Union = Type('Union', 'typing')
Type.Literal = Type('Literal', 'typing')
Type.TypeType = Type('Type', 'typing')
Type.Callable = Type('Callable', 'typing')
Type.ModuleType = Type('ModuleType', 'types')
Type.Any = Type('Any', 'typing').add_meta(unknown=True)
Type.overload = Type('@overload')

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
            return AnnotationType(self.state, qualname).add_meta(
                location=self.state.get_location(node))
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
            return AnnotationType(self.state, qualname).add_meta(
                location=self.state.get_location(node))
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
            is_callable = left.is_callable
            if isinstance(slicevalue, ast.Tuple):
                args: List[Type] = []
                for i,el in enumerate(slicevalue.elts):
                    if i==0 and is_callable and isinstance(el, ast.List):
                        args.extend(self._handle_list(el))
                    else:
                        args.append(self.visit(el))
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
        # transform Optional[x] into x | None 
        if left.is_optional:
            left = _union(*left.args, 'None').add_meta(**left.meta)
        return left.add_meta(location=self.state.get_location(node))

    def visit_BinOp(self, node: ast.BinOp) -> Type:
        # support new style unions
        if isinstance(node.op, ast.BitOr):
            left = self.visit(node.left)
            right = self.visit(node.right)
            return _union(left, right).add_meta(location=self.state.get_location(node))
        else:
            raise StaticValueError(node, 
                f"binary operation not supported: {node.op.__class__.__name__}",
                filename=self.state.get_filename(node)
            )

    def visit_Ellipsis(self, node: ast.Ellipsis | ast.Constant) -> Type:
        return Type("...").add_meta(
            location=self.state.get_location(node))

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
            return Type("None").add_meta(
                location=self.state.get_location(node))
        elif isinstance(value, type(...)):
            return self.visit_Ellipsis(node)
        if self.in_literal:
            return Type(repr(value)).add_meta(
                location=self.state.get_location(node))
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


    def _handle_list(self, node: ast.List) -> tuple[Type,...]:
        return tuple(self.visit(el) for el in node.elts)
        # ast.List is used in Callable, but we do not fully support it at the moment.
        # TODO: are the lists supposed to only allowed in callables?


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
            if __debug__:
                import traceback
                traceback.print_exc()
        except Exception as e:
            self._state.msg(
                f"unexpected {type(e).__name__} in type inference: {e}", ctx=expr
            )
            if __debug__:
                raise
        return None

    def builtin(self, name) -> Type:
        return BuiltinType(self._state, name)

    #########################################
    ###      expressions                  ###
    #########################################

    def visit_Constant(
        self, node: Union[ast.Constant, ast.Str, ast.NameConstant, 
                          ast.Bytes, ast.Num],
        path: list[ast.AST],
    ) -> Type:
        if isinstance(node, (ast.Str, ast.Bytes)):
            value: object = node.s
        elif isinstance(node, ast.Num):
            value = node.n
        else:
            value = node.value
        if value is None:
            return self.builtin("None").add_meta(
                location=self._state.get_location(node))
        return self.builtin(type(value).__name__).add_meta(
            location=self._state.get_location(node))
    
    visit_Str = visit_Constant
    visit_Bytes = visit_Constant
    visit_Num = visit_Constant
    visit_NameConstant = visit_Constant

    def visit_JoinedStr(self, node: ast.JoinedStr, path: list[ast.AST]) -> Type:
        return self.builtin("str").add_meta(
            location=self._state.get_location(node))

    def visit_List(self, node: ast.List | ast.Set, path: list[ast.AST]) -> Type:
        clsname = type(node).__name__.lower()
        subtype = Type.Any
        for element_node in node.elts:
            element_type = self.get_type(element_node, path)
            if element_type is None:
                return self.builtin(clsname)
            subtype = subtype.merge(element_type)
        if subtype.unknown:
            return self.builtin(clsname).add_meta(
                location=self._state.get_location(node))
        return self.builtin(clsname).add_args(args=(subtype,)).add_meta(
            location=self._state.get_location(node))

    visit_Set = visit_List

    def visit_Tuple(self, node: ast.Tuple, path: list[ast.AST]) -> Type:
        subtypes: tuple[Type, ...] = ()
        for element_node in node.elts:
            element_type = self.get_type(element_node, path)
            if element_type is None:
                return self.builtin("tuple").add_meta(
                    location=self._state.get_location(node))
            subtypes += (element_type,)
        if not subtypes:
            return self.builtin("tuple").add_meta(
                location=self._state.get_location(node))
        return self.builtin("tuple").add_args(args=subtypes).add_meta(
            location=self._state.get_location(node))

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
            return self.builtin("dict").add_meta(
                location=self._state.get_location(node))
        if keys_type.unknown:
            keys_type = Type.Any
        if values_type.unknown:
            values_type = Type.Any
        return self.builtin("dict").add_args(
            args=(keys_type, values_type)).add_meta(
            location=self._state.get_location(node))

    def visit_UnaryOp(self, node: ast.UnaryOp, path: list[ast.AST]) -> Type | None:
        if isinstance(node.op, ast.Not):
            return self.builtin("bool").add_meta(
                location=self._state.get_location(node))
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
                return self.builtin("float").add_meta(
                    location=self._state.get_location(node))
            return lt
        if lt.qualname in ("builtins.float", "builtins.int") and \
            rt.qualname in ("builtins.float", "builtins.int"):
            return self.builtin("float").add_meta(
                location=self._state.get_location(node))
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
        return result.add_meta(
            location=self._state.get_location(node))

    def visit_Compare(self, node: ast.Compare, path: list[ast.AST]) -> Type | None:
        if isinstance(node.ops[0], ast.Is):
            return self.builtin("bool").add_meta(
                location=self._state.get_location(node))
        # TODO: Use typeshed here to get precise type.
        return self.builtin("bool").add_meta(
            location=self._state.get_location(node))  # , ass={Ass.NO_COMP_OVERLOAD})

    def visit_ListComp(self, node: ast.ListComp, path: list[ast.AST]) -> Type | None:
        return self.builtin("list").add_meta(
            location=self._state.get_location(node))

    def visit_SetComp(self, node: ast.SetComp, path: list[ast.AST]) -> Type | None:
        return self.builtin("set").add_meta(
            location=self._state.get_location(node))

    def visit_DictComp(self, node: ast.DictComp, path: list[ast.AST]) -> Type | None:
        return self.builtin("dict").add_meta(
            location=self._state.get_location(node))

    def visit_GeneratorExp(
        self, node: ast.GeneratorExp, path: list[ast.AST]
    ) -> Type | None:
        return AnnotationType(self._state, 'typing.Iterator').add_meta(
            location=self._state.get_location(node))

    def visit_Call(self, node: ast.Call, path: list[ast.AST]) -> Type | None:
        assert node.func
        functype = self.get_type(node.func, path)
        if functype:
            if functype.is_type and len(functype.args) == 1:
                return functype.args[0].add_meta(
                    location=self._state.get_location(node))
            if functype.is_callable  or functype.is_overload:
                posargs = (self.get_type(n, path) or Type.Any for n in node.args)
                keywordargs = ((self.get_type(n.value, path) or Type.Any).add_meta(keyword=n.arg) for n in node.keywords)
                exprtype = Type.Callable.add_args(args=(*posargs, *keywordargs, TypeVariable()))
                try:
                    unified,_ = unify(functype, exprtype)
                except RuntimeError as e:
                    raise StaticValueError(node, f'Type unification failed: {e}', 
                                           filename=self._state.get_filename(node))

                rtype = unified.args[-1]
                if not rtype.unknown:
                    return rtype.add_meta(
                        location=self._state.get_location(node))
            
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
            return valuetype.add_meta(
                location=self._state.get_location(node))
        if valuetype.qualname in ("builtins.dict",) and len(valuetype.args) == 2:
            return valuetype.args[1].add_meta(
                location=self._state.get_location(node))
        if valuetype.qualname in ("builtins.list",) and len(valuetype.args) == 1:
            return valuetype.args[0].add_meta(
                location=self._state.get_location(node))
        if valuetype.qualname in ("builtins.tuple",):
            if len(valuetype.args) == 0:
                return None
            if len(valuetype.args) == 2 and valuetype.args[1].annotation == "...":
                return valuetype.args[0].add_meta(
                    location=self._state.get_location(node))
            try:
                indexvalue = self._state.literal_eval(node.slice)
            except StaticException:
                pass
            else:
                if not isinstance(indexvalue, int):
                    return None
                try:
                    return valuetype.args[indexvalue].add_meta(
                        location=self._state.get_location(node))
                except IndexError:
                    return None

            newtype = Type.Any
            for t in valuetype.args:
                newtype = newtype.merge(t)
            return newtype.add_meta(
                location=self._state.get_location(node))
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
            ann = _AnnotationToType(
                self._state, self._state.get_enclosing_scope(node)
            ).visit(assign.annotation)
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
        return newtype.add_meta(
            location=self._state.get_location(node))

    visit_alias = visit_Name_Load
    
    def visit_Attribute_Load(
        self, node: ast.Attribute, path: list[ast.AST]
    ) -> Type | None:
        valuetype = self.get_type(node.value, path)
        if valuetype is None:
            return None
        return self._get_type_attribute(valuetype, node.attr, node, path).add_meta(
            location=self._state.get_location(node))

    #########################################
    ###      statements                   ###
    #########################################

    def visit_arg(self, node: ast.arg, path: list[ast.AST], typevars:dict[str, Type]|None=None) -> Type | None:
        arg_def: Arg = self._state.get_def(node)  # type:ignore
        if arg_def.node.annotation is not None:
            try:
                annotation = self._replace_typevars(_AnnotationToType(
                    self._state, LazySeq(self._state.get_all_enclosing_scopes(node))[1]
                ).visit(arg_def.node.annotation), typevars=typevars or {})
            except StaticException:
                pass
            else:
                if not annotation.unknown:
                    if arg_def.kind == Parameter.VAR_POSITIONAL:
                        annotation = self.builtin("tuple").add_args(
                            args=[annotation, Type("...")])
                    if arg_def.kind == Parameter.VAR_KEYWORD:
                        annotation = self.builtin("dict").add_args(
                            args=[self.builtin("str"), annotation])
                    return annotation.add_meta(
                            location=self._state.get_location(node),
                            definition=arg_def)
        if (
            arg_def.default is not None
            and getattr(arg_def.default, "value", object()) is not None
        ):
            t = self.get_type(arg_def.default, path)
            if t:
                return t.add_meta(
                    location=self._state.get_location(node),
                    definition=self._state.get_def(node))
        if arg_def.kind == Parameter.VAR_POSITIONAL:
            return self.builtin("tuple").add_meta(
                location=self._state.get_location(node),
                definition=self._state.get_def(node))
        if arg_def.kind == Parameter.VAR_KEYWORD:
            return self.builtin("dict").add_args(
                args=[self.builtin("str"), Type.Any]).add_meta(
                location=self._state.get_location(node),
                definition=self._state.get_def(node))
        return None

    def visit_FunctionDef(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
        path: list[ast.AST],
        overloads: deque[Type]|None=None,
        typevars: dict[str, Type]|None=None,
    ) -> Type:

        typevars = typevars or {}
        if node.returns is not None:
            returntype = self._replace_typevars(_AnnotationToType(
                self._state, self._state.get_enclosing_scope(node)
            ).visit(node.returns), typevars)
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: better support for async functions
            returntype = AnnotationType(self._state, 'typing.Awaitable').add_args(args=(Type.Any,))
        else:
            returntype = Type.Any
        if any(
            self._state.expand_expr(n) in ("property", "builtins.property")
            for n in node.decorator_list
        ):
            return returntype
        
        argstypes: list[Type] = []
        if sys.version_info >= (3, 8):
            argstypes.extend(self.visit_arg(a, path, typevars) or Type.Any
                             for a in node.args.posonlyargs)
        argstypes.extend(self.visit_arg(a, path, typevars) or Type.Any for a in node.args.args)
        if node.args.vararg:
            t = self.visit_arg(node.args.vararg, path, typevars)
            if t and len(t.args)>1:
                argstypes.append(t.args[0].add_meta(**t.meta))
            elif t:
                argstypes.append(Type.Any.add_meta(**t.meta))
            else:
                argstypes.append(Type.Any)

        argstypes.extend((self.visit_arg(a, path, typevars) or Type.Any).add_meta(
            keyword=a.arg)  for a in node.args.kwonlyargs)
        if node.args.kwarg:
            t = self.visit_arg(node.args.kwarg, path, typevars)
            if t and len(t.args)>1:
                argstypes.append(t.args[1].add_meta(**t.meta))
            elif t:
                argstypes.append(Type.Any.add_meta(**t.meta))
            else:
                argstypes.append(Type.Any)

        functiontype = Type.Callable.add_args(args=(*argstypes, returntype)).add_meta(
            qualname=self._state.get_qualname(node),
            location=self._state.get_location(node), 
            definition=self._state.get_def(node),
        )
        # check for overloads
        overloads = overloads or deque()
        overloads.appendleft(functiontype)

        left_sibling = self._state.get_sibling(node, direction=-1)
        if isinstance(left_sibling, ast.FunctionDef) and left_sibling.name == node.name and (any(
            self._state.expand_expr(n) in ("typing.overload", "typing_extensions.overload") for n 
            in left_sibling.decorator_list) or self._state.get_root(node).name() in ('typing', 'builtins')):
                self.visit_FunctionDef(left_sibling, path, overloads, typevars)
        
        if len(overloads)>1:
            return Type.overload.add_args(args=overloads)
        return functiontype

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Module(self, node: ast.Module, path: list[ast.AST]) -> Type:
        modname = self._state.get_qualname(node)
        return Type.ModuleType.add_meta(
            qualname=modname, 
            location=self._state.get_location(node), 
            definition=self._state.get_def(node),)

    def visit_ClassDef(self, node: ast.ClassDef, path: list[ast.AST]) -> Type:
        return Type.TypeType.add_args(args=[
            ClsType(self._state, self._state.get_def(node)) ]).add_meta(
                # for consistency, we set all 3 meta fields even if information
                # is duplicated inside args[0].
                qualname=self._state.get_qualname(node), 
                location=self._state.get_location(node), 
                definition=self._state.get_def(node),)
          
    #########################################
    ###      type inference helpers       ###
    #########################################

    def _get_type_attribute(
        self, valuetype: Type, 
        attr: str, 
        ctx: ast.AST, 
        path: list[ast.AST]
    ) -> Type:
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
            except StaticException as e:
                print(e)
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
            node = definition.node
            attrtype = self.get_type(node, path)
            if attrtype is not None:
                # remove 'self' argument on bound methods.
                if (isinstance(node, (ast.FunctionDef, 
                                    ast.AsyncFunctionDef)) and
                        is_instance_method(node) and 
                        not valuetype.is_type and 
                        attrtype.is_callable):
                    attrtype = attrtype._replace(args=attrtype.args[1:])

                newtype = newtype.merge(attrtype)
        
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
    
    def _replace_typevars(self, type: Type, typevars:dict[str, Type]) -> Type:
        """
        Replace raw typevars with `TypeVariable` types.
        """
        # lectures about unification of typevars: 
        # https://stackoverflow.com/questions/65362422/type-unification-algorithm-in-python-how-to-reject-unifya-b-int-int
        # https://gist.github.com/dhilst/b5b198af93302ade61ccbfe3b094621a
        # https://eli.thegreenplace.net/2018/unification/
        # https://github.com/caterinaurban/Typpete/blob/master/typpete/src/annotation_resolver.py
        # https://github.com/pfalcon/picompile
        # https://github.com/eliphatfs/typhon/tree/master/typhon/core/type_system/intrinsics
        # https://github.com/serge-sans-paille/tog/blob/master/tog.py#L391
        class NotATypeVar(Exception):
            ...
        
        try:
            definition = type.definition
            if not isinstance(definition, Var):
                raise NotATypeVar()
            assign = self._state.get_parent(definition)
            if not isinstance(assign, ast.Assign):
                raise NotATypeVar()
            if not isinstance(assign.value, ast.Call):
                raise NotATypeVar()
            name = node2dottedname(assign.value.func)
            if not name:
                raise NotATypeVar()
            if self._state.expand_name(self._state.get_root(definition), 
                                    '.'.join(name)) == 'typing.TypeVar':
                try:
                    tv = typevars[definition.name()]
                    return type._replace(name=tv.name)
                except KeyError:
                    # preserve meta data
                    tv = type._replace(name=TypeVariable().name)
                    typevars[definition.name()] = tv
                    return tv
        except NotATypeVar:
            pass
        
        subtypes = [self._replace_typevars(t, typevars) for t in type.args]
        if all(s.unknown for s in subtypes):
            subtypes = []
        
        return type._replace(args=subtypes)
    
    # def _get_typedef(self, typ:Type) -> Def:
    #     """
    #     Find the definition of a Type.
    #     """
    #     # supports classes, modules, functions and variables at the moment.
    #     location:NodeLocation|None
    #     qualname:str
    #     if typ.is_module:
    #         qualname = typ.get_meta('qualname', str) # type:ignore[assignment]
    #         if qualname is None:
    #             raise StaticValueError(typ, "no module definition")
    #         hint:type[Def]|tuple[type[Def],...] = Mod
    #         location = typ.get_meta('location', NodeLocation)
    #     elif typ.is_callable:
    #         qualname = typ.get_meta('qualname', str) # type:ignore[assignment]
    #         if qualname is None:
    #             raise StaticValueError(typ, "no function definition")
    #         hint = Func
    #         location = typ.get_meta('location', NodeLocation)
    #     else:
    #         location = typ.location
    #         qualname = typ.qualname
    #         if '.' not in qualname:
    #             raise StaticValueError(typ, f"won't find anything for type {typ}")
    #         hint = (Cls, Var)
    #     if hint is Mod:
    #         m = self._state.get_module(qualname)
    #         if m is None:
    #             raise StaticValueError(typ, 
    #                 f"unknown module {typ.name!r}",)
    #         return m
    #     else:
    #         return self._find_typedef(qualname, hint=hint, location=location)

    def _flatten_typedefs(
        self, valuetype: Type, seen: set[Type]
    ) -> Iterator[Tuple[Type, Def | None]]:
        """
        Get the definition of each resolvable 
        top-level types in this type instance.
        Unwrap unions and overloads.
        """
        if valuetype in seen:
            return
        seen.add(valuetype)

        if valuetype.is_union or valuetype.is_overload:
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
                        self._state.msg(f'incomplete {valuetype.name}: {e.location()}:{e.msg()}', 
                                        ctx=valuetype.get_meta('location', NodeLocation))

        elif valuetype.is_literal:
            # unwrap_Literal_classdef
            try:
                val = ast.literal_eval(valuetype.args[0].name)
            except Exception:
                return
            if val is None:
                yield valuetype, None
            else:
                definition = self.builtin(type(val).__name__).definition
                # if definition is None:
                #     raise StaticNameError(type(val).__name__, filename='<missing builtin>')
                yield valuetype, definition
        elif valuetype.is_type or valuetype.is_module or valuetype.is_callable:
            definition = valuetype.get_meta('definition', Def)
            yield valuetype, definition
        else:
            # catch-all case
            # assume it's an instance of a type in the project
            definition = valuetype.definition
            # if definition is None:
            #     raise StaticNameError(valuetype.qualname, filename='<missing>')
            yield valuetype, definition

class _TypeVariableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        if (isinstance(instance, Type) and instance.is_typevar):
            return True
        return super().__instancecheck__(instance)
    
class TypeVariable(Type, metaclass=_TypeVariableMeta):
    """
    >>> tv = TypeVariable()
    >>> assert isinstance(tv, TypeVariable)
    >>> repr(tv)
    "Type(name='@TypeVar...', scope='', args=(), meta=...)"
    >>> assert tv != TypeVariable()
    >>> assert Type('@TypeVar43')==Type('@TypeVar43')
    """
    _id = 0
    def __new__(cls) -> Type: # type:ignore
        TypeVariable._id += 1
        return Type(f'@TypeVar{TypeVariable._id}')
    @classmethod
    def _reset(cls) -> None:
        cls._id = 0

#### Type variable support
# Copyright (c) 2016, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def prune(t:Type) -> Type:
    # no need to prune frozen instances.
    return t

def occurs_in_type(v:Type, type2:Type) -> bool:
    """Checks whether a type variable occurs in a type expression.

    Note: Must be called with v pre-pruned

    Args:
        v:  The TypeVariable to be tested for
        type2: The type in which to search

    Returns:
        True if v occurs in type2, otherwise False
    """
    pruned_type2 = prune(type2)
    if pruned_type2 == v:
        return True
    return occurs_in(v, pruned_type2.args)

def occurs_in(t:Type, types:Sequence[Type]) -> bool:
    """Checks whether a types variable occurs in any other types.

    Args:
        t:  The TypeVariable to be tested for
        types: The sequence of types in which to search

    Returns:
        True if t occurs in any of types, otherwise False
    """
    return any(occurs_in_type(t, t2) for t2 in types)


def _approximate_overload(b:Type, types:Sequence[Type], subst:Subst) -> Type:
    def try_unify(t:Type, ts:Sequence[Type]) -> Type|None:
        if isinstance(t, TypeVariable):
            return None
        if any(isinstance(tp, TypeVariable) for tp in ts):
            return None
        # overapproximate 't'
        new_args = list(t.args)
        for i, tt in enumerate(t.args):
            its = [prune(tp.args[i]) for tp in ts]
            if any(isinstance(it, TypeVariable) for it in its):
                continue
            it0 = its[0]
            it0ntypes = len(it0.args)
            if all(((it.name == it0.name) and (len(it.args) == it0ntypes)) for it in its):
                ntypes = [TypeVariable() for _ in range(it0ntypes)]
                new_tt = it0._replace(args=ntypes)
                # new_tt.__class__ = it0.__class__
                tt,_ = unify(tt, new_tt, subst)
                new_args[i] = try_unify(prune(tt), [prune(it) for it in its]) or tt
        return t._replace(args=new_args)
    
    r = try_unify(b, types)
    if r is None:
        raise RuntimeError("Not unified {} and {}, overload approximation failed".format(types, b.annotation))
    return r

def better_signature(callable_type:Type) -> inspect.Signature | None:
    """
    Get a signature from the function def information wrapped by this type instance.
    Return None if there is no function wrapped, in the case of a written 'Callable' annotation for instance.
    """
    definition = callable_type.get_meta('definition', Def)
    if not isinstance(definition, Func):
        return None
        raise RuntimeError(f'missing definition of callable {callable_type.annotation}')
    parameters: list[inspect.Parameter] = []
    for i, arg in enumerate(callable_type.args[:-1]):
        argdef = arg.get_meta('definition', Arg)
        if argdef is None:
            # the type vars seem to replace the type here:/ with no informations anymore.
            raise RuntimeError(f'missing definition of callable {definition.name()!r} argument {i}: {arg}')
        parameters.append(argdef.to_parameter().replace(annotation=arg))
    return inspect.Signature(parameters, return_annotation=callable_type.args[-1])

def callable_struct(callable_type:Type) -> Tuple[Sequence[Type], Mapping[str, Type], Type]:
    """
    Returns tuple: (args, kwargs, rtype)
    """
    if not callable_type.args:
        return [], {}, Type.Any

    positionals: list[Type] = []
    keywords: dict[str, Type] = {}

    for a in callable_type.args[:-1]:
        keyword = a.get_meta('keyword', str)
        if keyword is not None:
            keywords[keyword] = keywords.setdefault(keyword, Type.Any).merge(a)
        else:
            if keywords:
                raise RuntimeError('invalid callable, keywords came before positionals :/')
            positionals.append(a)
    
    return positionals, keywords, callable_type.args[-1]

def poor_signature(callable_type:Type) -> inspect.Signature:
    """
    Get a approximated signature based on a Callable type 
    (created from annotations or from ast.Call before unification). 
    """
    positionals, keywords, rtype = callable_struct(callable_type)
    # simplify signatures
    parameters: list[inspect.Parameter] = []
    parameters.extend((inspect.Parameter(f'__{i}__', 
                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       annotation=t) for i,t in enumerate(positionals)))
    parameters.extend((inspect.Parameter(name,
                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       annotation=t) for name,t in keywords.items()))
    return inspect.Signature(parameters, return_annotation=rtype)

def bind_callables(a:Type, b:Type) -> Tuple[Type, Type, inspect.BoundArguments]:
    """
    Dertemine signature based compatibility between two callables.
    Returns (caller, callee, bound arguments).
    """
    def _add_err_note(e:Exception):
        if sys.version_info >= (3, 11):
            shortsig = inspect.Signature(
                [p.replace(annotation=p.annotation.annotation,
                            default=inspect.Parameter.empty,
                            ) for p in sig.parameters.values()])
            
            e.add_note(f'signature={str(shortsig)}')
            e.add_note(f'args={[a.annotation for a in args]}')
            e.add_note(f'keywords={dict((n,a.annotation) for n,a in keywords.items())}')

    caller, callee = a, b
    sig = better_signature(callee) or poor_signature(callee)
    args, keywords, _ = callable_struct(caller)
    assert len(args)+len(keywords)+1==len(caller.args)
    try:
        bargs = sig.bind(*args, **keywords)
    except Exception as e:
        _add_err_note(e)
        # raise RuntimeError("Callable type mismatch: {0} != {1}".format(a.annotation, b.annotation)) from e

        caller, callee = b, a
        sig = better_signature(callee) or poor_signature(callee)
        args, keywords, _ = callable_struct(caller)
        assert len(args)+len(keywords)+1==len(caller.args)

        try:
            bargs = sig.bind(*args, **keywords)
        except Exception as e:
            _add_err_note(e)
            raise RuntimeError("Callable type mismatch: {0} != {1}".format(a.annotation, b.annotation)) from e
    
    return caller, callee, bargs

Subst = frozenset[tuple[str, Type]]

def cleanup_unresolved_typevars(term:Type | None) -> Type | None:
    if term is None:
        return None
    if term.is_typevar:
        return Type.Any
    args = [cleanup_unresolved_typevars(t) for t in term.args]
    if all(s.unknown for s in args):
        args = []
    return term._replace(args=args)

def subst1(term: Type, subst: Subst) -> Tuple[Type, Subst]:
    if term.is_typevar:
        _replacements: list[Type] = []
        # Same note as `substmult`.
        # we should be better with a mapping data strucutre instead.
        for name, replacement in reversed(subst):
            if name == term.name:
                _replacements.append(replacement)
        if _replacements:
            t = Type.Any
            for r in _replacements:
                # maybe merge-unify instead of this?
                try:
                    t, s = unify(t, r, subst)
                except RuntimeError:
                    # two possible values for a given type variable doesn't unify. 
                    # we then use the first common supertype, that's not object.
                    supertype = r.supertype
                    while 1:
                        if supertype is None:
                            raise
                        if supertype.qualname == 'builtins.object':
                            raise
                        if supertype.supertype_of(t):
                            break
                        supertype = supertype.supertype
                    supertype = supertype._replace(args=r.args)
                    t, s = unify(t, supertype, subst)
                subst |= s
            return t, subst
        else:
            return term, subst
    else:
        new_args = []
        for arg in term.args:
            t, s = subst1(arg, subst)
            new_args.append(t)
            subst |= s
        return term._replace(args=new_args), subst

def substmult(subst: Subst, replacement: Subst) -> Subst:
    """
    Apply substitutions to all elements of the subst with content in replacement.
    """
    # I feel like this function drastically increased the algorithmic complexity of
    # our type inference. Something should be done about it, but first we needs more tests.
    new_subst = ordered_set()
    try:
        for (name, term) in subst:
            t, s = subst1(term, replacement)
            v = (name, t)
            new_subst.add(v)
            new_subst.update(s)
        return new_subst
    except Exception as e:
        if sys.version_info >= (3,11):
            e.add_note(f'subst={subst}')
        raise

def unify(t1:Type, t2:Type, subst:Subst=None) -> Tuple[Type, Subst]:
    """Unify the two types t1 and t2.

    Returns a new type that represent the unification of both type.

    Args:
        t1: The first type to be made equivalent
        t2: The second type to be be equivalent
        subst: type variables substitutions.

    Returns:
        Tuple: (The unified type, Subst)

    Raises:
        InferenceError: Raised if the types cannot be unified.
    
    >>> t = Type.overload.add_args(args=
    ... [Type.Callable.add_args([Type('float', 'builtins'), Type('int', 'builtins'),]), 
    ...  Type.Callable.add_args([Type.Union.add_args([Type('str', 'builtins'), 
    ...                                               Type('bytes', 'builtins'),
    ...                                               Type('object', 'builtins')]), 
    ...                          Type('str', 'builtins'),])])
    >>> expr1 = Type.Callable.add_args([Type('int', 'builtins'), TypeVariable()])
    >>> print(t.annotation)
    (float) -> int | (str | bytes | object) -> str
    >>> print(expr1.annotation)
    (int) -> @TypeVar...
    >>> print(unify(expr1, t)[0].annotation)
    (float) -> int
    >>> expr2 = Type.Callable.add_args([Type('float', 'builtins'), TypeVariable()])
    >>> print(unify(expr2, t)[0].annotation)
    (float) -> int
    >>> expr3 = Type.Callable.add_args([Type('bytes', 'builtins'), TypeVariable()])
    >>> print(unify(expr3, t)[0].annotation)
    (object) -> str
    >>> t2 = Type.Callable.add_args([Type('list', 'builtins').add_args(
    ...     [Type('@TypeVar1')]), Type('list', 'builtins').add_args(
    ...     [Type('@TypeVar1')]),])
    >>> expr4 = Type.Callable.add_args([Type('list', 'builtins').add_args([Type('float', 'builtins')]), Type('@TypeVar2')])
    >>> print(unify(expr4, t2)[0].annotation)
    (list[float]) -> list[float]
    """
    # basically a mix a these two approches, with support for subtyping as well.
    # https://gist.github.com/dhilst/b5b198af93302ade61ccbfe3b094621a
    # https://github.com/serge-sans-paille/tog/blob/master/tog.py
    subst = ordered_set() if subst is None else subst

    if t1 == t2:
        return t1, subst
    elif t2.is_typevar and not t1.is_typevar:
        return unify(t2, t1, subst)
    elif t1.is_typevar:
        if occurs_in_type(t1, t2):
            raise RuntimeError("recursive unification")
        newsubst: Subst = ordered_set([(t1.name, t2)])
        # substitutions stucture
        subst = substmult(subst, newsubst) | newsubst # type: ignore
        # t2, subst = 
        return subst1(t1, subst)
    elif t2.is_overload and not t1.is_overload:
        return subst1(*unify(t2, t1, subst))
    elif t1.is_overload:
        types: list[Tuple[Type, Subst]] = []
        errs: list[Exception] = []
        for t in t1.args:
            try:
                types.append(unify(t, t2, subst))
            except RecursionError: 
                raise
            except RuntimeError as e:
                errs.append(e)
        if types:
            if len(types) == 1:
                return subst1(*unify(types[0][0], t2, types[0][1]))
            # too many overloads are found, so extract as many 
            # information as we can, and unify the rest with the first overload match.
            return subst1(*unify(types[0][0], 
                         _approximate_overload(t2, [_t[0] for _t in types], types[0][1]), 
                         types[0][1]))
        else:
            raise RuntimeError("Type mismatch, no overload found: {} and {}: {}".format(t1.annotation, t2.annotation, errs))
    elif t2.is_union and not t1.is_union:
        return subst1(*unify(t2, t1, subst))
    elif t1.is_union:
        types = []
        errs = []
        for t in t1.args:
            try:
                types.append(subst1(*unify(t, t2, subst)))
            except RecursionError: 
                raise 
            except RuntimeError as e:
                errs.append(e)
        if types:
            new_type = Type.Any
            for t,s in types:
                # Maybe we're doing too much of computation here?
                # subst |= s
                # t, s = subst1(*unify(t, t2, subst))
                new_type = new_type.merge(t)
                # new_type = new_type.merge(t)
                subst |= s
            return new_type, subst
        else:
            raise RuntimeError("Type mismatch, none of the union element matches: {} and {}: {}".format(t1.annotation, t2.annotation, errs))
    elif t2.unknown:
        return subst1(t1, subst)
    elif t1.unknown:
        return subst1(t2, subst)
    elif t2.is_callable and t1.is_callable:
        caller, callee, bargs = bind_callables(t1, t2)
        # now let's adjust callee's arguments to match caller's
        # transfer typevars of course!
        new_args = [*(bargs.signature.parameters[a].annotation 
                      for a in bargs.arguments), callee.args[-1]]

        if new_args != list(callee.args):
            callee = callee._replace(args=new_args)

        assert len(callee.args) == len(caller.args), f'{callee.args} != {caller.args}'
        
        # Unify arguments
        unified_args: list[Type] = []
        for p, q in zip(callee.args, caller.args):
            t, s = subst1(*unify(p, q, subst))
            subst |= s
            unified_args.append(t)
        
        return callee._replace(args=unified_args), subst
    
    elif len(t1.args) != len(t2.args):
        raise RuntimeError("Type mismatch: {0} != {1}".format(t1.annotation, t2.annotation))
   
    # catch-all case.
    if t2.qualname == t1.qualname or t1.supertype_of(t2):
        keep = t1
    elif t2.supertype_of(t1):
        keep = t2
    else:
        raise RuntimeError("Type mismatch: {0} does not match {1}".format(t1.qualname, t2.qualname))
    
    # Unify arguments
    unified_args = []
    for p, q in zip(t1.args, t2.args):
        t, s = subst1(*unify(p, q, subst))
        unified_args.append(t)
        subst |= s
    
    return keep._replace(args=unified_args), subst

if __debug__:
    unify_org = unify
    def debug_unify(t1:Type, t2:Type, subst:Subst=None) -> Tuple[Type, Subst]:
        subst = ordered_set() if subst is None else subst

        subst_before = ordered_set(subst)
        try:
            r,s = unify_org(t1, t2, subst)
        except RuntimeError:
            print(f'Unification of:\n- {t1.annotation}\n- {t2.annotation}\n=> MISMATCH\n Typevars before: {[(n,t.annotation) for n,t in subst_before]})')
            raise
        else:
            print(f'Unification of:\n- {t1.annotation}\n- {t2.annotation}\n=> {r.annotation}')
                  # Typevars before: {[(n,t.annotation) for n,t in subst]})\n Typevars after: {[(n,t.annotation) for n,t in s]}\n')
            if not set(s).issuperset(subst_before):
                print(f'Removals: Diff: {[(n,t.annotation) for n,t in set(subst_before).difference(s)]}')
            if set(s).difference(subst_before):
                print(f'Additions: Diff: {[(n,t.annotation) for n,t in set(s).difference(subst_before)]}')
            return r,s
    unify = debug_unify