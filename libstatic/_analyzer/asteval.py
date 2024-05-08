from __future__ import annotations

import ast
import numbers
import sys
import types
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Type,
    Union,
    overload,
    TypeVar,
    Generic,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from typing import TypeAlias
    from .state import _MinimalState

from .._lib.shared import node2dottedname
from .._lib.assignment import get_stored_value
from .._lib.exceptions import (
    StaticValueError,
    StaticCodeUnsupported,
    StaticException,
    StaticTypeError,
    StaticAmbiguity,
    StaticEvaluationError,
    StaticUnknownValue,
)

_AST_SEQUENCE_TO_TYPE: ( # type:ignore[type-arg]
    "dict[type[ast.Tuple]|type[ast.List]|type[ast.Set],type[tuple]|type[list]|type[set]]" 
) = {
    ast.Tuple: tuple,
    ast.List: list,
    ast.Set: set,
}

LiteralValue: "TypeAlias" = (
    "list[LiteralValue]|set[LiteralValue]|tuple[LiteralValue,...]|"
    "dict[LiteralValue, LiteralValue]|"
    "str|numbers.Number|bool|bytes|None|slice|types.EllipsisType"
)
ASTOrLiteralValue = Union[LiteralValue, ast.AST]

T = TypeVar('T')
class _EvalBaseVisitor(Generic[T]):

    _MAX_JUMPS = int(sys.getrecursionlimit() / 1.5)
    # each visited ast node counts for one jump.

    def __init__(self, state: '_MinimalState') -> None:
        self._state = state
    
    def _visit(self, node: ast.AST, path: List[ast.AST]) -> T:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        try:
            visitor = getattr(self, method)
        except AttributeError:
            raise StaticCodeUnsupported(node, "node type", 
                                        filename=self._state.get_filename(node))
        else:
            return visitor(node, path) # type: ignore
    
    def _visit_ctx_dependent(
        self, node: Union[ast.Attribute, ast.Name], path: List[ast.AST]
    ) -> T:
        ctx = getattr(node, "ctx", ast.Load())
        ctxname = ctx.__class__.__name__
        method = f"visit_{node.__class__.__name__}_{ctxname}"
        try:
            visitor = getattr(self, method)
        except AttributeError:
            raise StaticCodeUnsupported(node, f"node ctx {ctxname}", 
                                        filename=self._state.get_filename(node))
        else:
            return visitor(node, path) # type: ignore
    
    visit_Name = visit_Attribute = _visit_ctx_dependent

    def visit(self, node: Any, path: List[ast.AST]) -> T:
        if node in path:
            raise StaticValueError(node, f"node has cyclic definition, "
                                   f'path={[str(self._state.get_location(n)) for n in path]+[str(self._state.get_location(node))]}', # type:ignore
                                   filename=self._state.get_filename(node))
        if len(path) > self._MAX_JUMPS:
            raise StaticCodeUnsupported(node, "expression is too complex", 
                                        filename=self._state.get_filename(node))
        # fork path
        path = path.copy()
        path.append(node)
        return self._visit(node, path)

class _ASTEval(_EvalBaseVisitor[ASTOrLiteralValue]):
    # Custom implementation of NodeVisitor which accept one extra argument
    # to it's visti() method.

    def __init__(self, state: '_MinimalState', raise_on_ambiguity: bool = False) -> None:
        super().__init__(state)
        self._raise_on_ambiguity = raise_on_ambiguity

    def _returns(self, ob: ast.stmt, _: Any) -> ast.stmt:
        return ob

    visit_Module = visit_ClassDef = visit_FunctionDef = \
        visit_AsyncFunctionDef = visit_Lambda = _returns

    def visit_Attribute_Load(
        self, node: ast.Attribute, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        namespace = self.visit(node.value, path)
        if isinstance(namespace, (ast.Module, ast.ClassDef)):
            attribs = self._state.get_attribute(namespace, node.attr)
            if len(attribs) > 1 and self._raise_on_ambiguity:
                raise StaticAmbiguity(
                    node, f"{len(attribs)} potential definitions found",
                    filename=self._state.get_filename(node)
                )
            return self.visit(attribs[-1].node, path)
        else:
            raise StaticTypeError(namespace, expected="Module or ClassDef", 
                                  filename=self._state.get_filename(node))

    def visit_Name_Load(self, node: ast.Name, path: List[ast.AST]) -> ASTOrLiteralValue:
        # TODO: integrate with reachability analysis
        # Use goto to compute the value of this symbol
        name_def = self._state.goto_def(node,
                        raise_on_ambiguity=self._raise_on_ambiguity)
        return self.visit(name_def.node, path)

    def visit_Name_Store(
        self, node: ast.Name, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        # doesn't support augmented assignments, for loops target, etc..
        assign = self._state.get_parent_instance(
            node, (ast.Assign, ast.AnnAssign, ast.AugAssign)
        )
        if isinstance(assign, ast.AugAssign):
            raise StaticCodeUnsupported(assign, "augmented assignments", 
                                        filename=self._state.get_filename(node))
        value = get_stored_value(node, assign=assign) # type: ignore
        if value is not None:
            return self.visit(value, path)
        else:
            raise StaticValueError(node, "node has no value")


class _GotoDefinition(_ASTEval):
    def __init__(
        self,
        state: '_MinimalState',
        raise_on_ambiguity: bool = False,
        follow_aliases: bool = False,
        follow_imports: bool = True,
    ) -> None:
        super().__init__(state, raise_on_ambiguity)

        if follow_aliases:
            self.visit_Name_Store = self.visit_Name_FollowAliases #type:ignore
        if not follow_imports:
            self.visit_alias = self.visit_alias_DontFollowImports #type:ignore

    def visit(self, node: Any, path: List[ast.AST]) -> ast.AST:
        v = super().visit(node, path)
        if not isinstance(v, ast.AST):
            raise StaticTypeError(v, expected="definition", 
                                  filename=self._state.get_filename(path[-1]))
        return v

    def visit_alias(
        self: _ASTEval, node: ast.alias, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        name_def = self._state.goto_def(node, 
                    raise_on_ambiguity=self._raise_on_ambiguity)
        
        return self.visit(name_def.node, path)

    def visit_alias_DontFollowImports(
        self, node: ast.alias, path: List[ast.AST]
    ) -> ast.alias:
        return node

    def visit_Name_Store(self, node: ast.Name, path: List[ast.AST]) -> ast.Name:
        return node

    def visit_Name_FollowAliases(
        self, node: ast.Name, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        # doesn't support augmented assignments
        # TODO: refactor this code
        try:
            assign = self._state.get_parent_instance(node, (ast.Assign, ast.AnnAssign))
        except StaticException as e:
            raise StaticCodeUnsupported(node, "name", filename=self._state.get_filename(node)) from e
        value = get_stored_value(node, assign=assign) # type:ignore[arg-type]
        if node2dottedname(value) is not None:
            # it's an alias, so follow-it
            return self.visit(value, path)
        else:
            return node


# TODO: write a subclass of GotoDefinition that works with manually created nodes
# like unstringed annotations or nodes created in the infer-type function.

# Safe versions of functions to prevent denial of service issues
MAX_EXPONENT = 10000
MAX_STR_LEN = 2 << 17  # 256KiB
MAX_SHIFT = 1000


def safe_pow(base: Any, exp: Any) -> Any:
    """safe version of pow"""
    if isinstance(exp, numbers.Number):
        if exp > MAX_EXPONENT:  # type: ignore
            raise RuntimeError(f"Invalid exponent, max exponent is {MAX_EXPONENT}")
    return base**exp


def safe_mult(arg1: Any, arg2: Any) -> Any:
    """safe version of multiply"""
    if (
        isinstance(arg1, str)
        and isinstance(arg2, int)
        and len(arg1) * arg2 > MAX_STR_LEN
    ):
        raise RuntimeError(
            f"String length exceeded, max string length is {MAX_STR_LEN}"
        )
    return arg1 * arg2


def safe_add(arg1: Any, arg2: Any) -> Any:
    """safe version of add"""
    if (
        isinstance(arg1, str)
        and isinstance(arg2, str)
        and len(arg1) + len(arg2) > MAX_STR_LEN
    ):
        raise RuntimeError(
            f"String length exceeded, max string length is {MAX_STR_LEN}"
        )
    return arg1 + arg2


def safe_lshift(arg1: Any, arg2: Any) -> Any:
    """safe version of lshift"""
    if isinstance(arg2, numbers.Number):
        if arg2 > MAX_SHIFT:  # type: ignore
            raise RuntimeError(f"Invalid left shift, max left shift is {MAX_SHIFT}")
    return arg1 << arg2


# TODO: use the operator module for perf
_OPERATORS: Mapping[str, Callable[[LiteralValue, LiteralValue], LiteralValue]] = {
    "Is": lambda a, b: a is b,
    "IsNot": lambda a, b: a is not b,
    "In": lambda a, b: a in b,  # type: ignore
    "NotIn": lambda a, b: a not in b,  # type: ignore
    "Add": safe_add,
    "BitAnd": lambda a, b: a & b,  # type: ignore
    "BitOr": lambda a, b: a | b,  # type: ignore
    "BitXor": lambda a, b: a ^ b,  # type: ignore
    "Div": lambda a, b: a / b,  # type: ignore
    "FloorDiv": lambda a, b: a // b,  # type: ignore
    "LShift": safe_lshift,
    "RShift": lambda a, b: a >> b,  # type: ignore
    "Mult": safe_mult,
    "Pow": safe_pow,
    "MatMult": lambda a, b: a @ b,  # type: ignore
    "Sub": lambda a, b: a - b,  # type: ignore
    "Mod": lambda a, b: a % b,  # type: ignore
    "And": lambda a, b: a and b,
    "Or": lambda a, b: a or b,
    "Eq": lambda a, b: a == b,
    "Gt": lambda a, b: a > b,  # type: ignore
    "GtE": lambda a, b: a >= b,  # type: ignore
    "Lt": lambda a, b: a < b,  # type: ignore
    "LtE": lambda a, b: a <= b,  # type: ignore
    "NotEq": lambda a, b: a != b,
}

_UNARY_OPERATORS: Mapping[str, Callable[[LiteralValue], LiteralValue]] = {
    "Invert": lambda a: ~a,  # type: ignore
    "Not": lambda a: not a,  # type: ignore
    "UAdd": lambda a: +a, # type: ignore
    "USub": lambda a: -a,  # type: ignore
}


@overload
def op2func(
    oper: ast.AST, nodetype: Union[Type[ast.BinOp], Type[ast.Compare]]
) -> Callable[[LiteralValue, LiteralValue], LiteralValue]:
    ...


@overload
def op2func(
    oper: ast.AST, nodetype: Type[ast.UnaryOp]
) -> Callable[[LiteralValue], LiteralValue]:
    ...


def op2func(
    oper: ast.AST,
    nodetype: Union[Type[ast.BinOp], Type[ast.Compare], Type[ast.UnaryOp]],
) -> Union[
    Callable[[LiteralValue, LiteralValue], LiteralValue],
    Callable[[LiteralValue], LiteralValue],
]:
    """Return function for operator nodes."""
    if issubclass(nodetype, ast.UnaryOp):
        return _UNARY_OPERATORS[oper.__class__.__name__]
    return _OPERATORS[oper.__class__.__name__]


def ensure_literal(o: ASTOrLiteralValue) -> LiteralValue:
    if isinstance(o, ast.AST):
        raise StaticTypeError(o, expected="literal")
    return o

# this class is inspired by several projects:
# simpleeval, asteval and typeshed_client
class _LiteralEval(_ASTEval):
    def __init__(
        self,
        state: '_MinimalState',
        *,
        known_values: Mapping[str, LiteralValue],
        raise_on_ambiguity: bool = False,
        follow_imports: bool = False,
    ) -> None:
        super().__init__(state, raise_on_ambiguity)
        self._known_values = known_values

        if follow_imports:
            self.visit_alias = self.visit_alias_FollowImports # type:ignore

    def visit_alias_Default(self, node: ast.alias, path: List[ast.AST]) -> LiteralValue:
        fullname = self._state.get_def(node).target()
        if fullname in self._known_values:
            return self._known_values[fullname]
        raise StaticUnknownValue(node, fullname, 
                                 filename=self._state.get_filename(node))

    def visit_alias_FollowImports(
        self, node: ast.alias, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        try:
            return self.visit_alias_Default(node, path)
        except StaticUnknownValue:
            return _GotoDefinition.visit_alias(self, node, path)

    visit_alias = visit_alias_Default

    def visit_Name_Load(self, node: ast.Name, path: List[ast.AST]) -> ASTOrLiteralValue:
        # check if this name is part of the known values
        if node.id in self._known_values:
            return self._known_values[node.id]
        fullname = self._state.expand_expr(node)
        if fullname is not None and fullname in self._known_values:
            return self._known_values[fullname]
        return super().visit_Name_Load(node, path)

    def visit_Attribute_Load(
        self, node: ast.Attribute, path: List[ast.AST]
    ) -> ASTOrLiteralValue:
        # check if this name is part of the known values
        fullname = self._state.expand_expr(node)
        if fullname is not None and fullname in self._known_values:
            return self._known_values[fullname]
        return super().visit_Attribute_Load(node, path)

    def visit_Constant(self, node: ast.Constant, path: List[ast.AST]) -> LiteralValue:
        v = node.value
        if not isinstance(v, (str, numbers.Number, bool, bytes, 
                              type(None), type(...))):
            raise StaticTypeError(node.value, expected="literal", 
                                  filename=self._state.get_filename(node))
        return v

    def visit_List(
        self, node: Union[ast.Tuple, ast.List, ast.Set], path: List[ast.AST]
    ) -> LiteralValue:
        ctx = getattr(node, "ctx", ast.Load())
        if not isinstance(ctx, ast.Load):
            raise StaticTypeError(ctx, expected="Load", filename=self._state.get_filename(node))
        # evaluate the list elements
        values: List[LiteralValue] = []
        for idx, elt in enumerate(node.elts):
            try:
                if isinstance(elt, ast.Starred):
                    v = ensure_literal(self.visit(elt.value, path))
                    if not isinstance(v, (list, tuple, set, dict, str, bytes)):
                        raise StaticTypeError(v, expected="starred value iterable", 
                                              filename=self._state.get_filename(node))
                    else:
                        values.extend(v) # type: ignore[arg-type]
                else:
                    values.append(ensure_literal(self.visit(elt, path)))

            except StaticException as e:
                # we make sure that we don't fail the
                # evaluation of the whole list just because
                # some elements are failing to be evaluated.
                msg = f"Cannot evaluate {type(node).__name__.lower()} element at index {idx}: {e}"
                if self._raise_on_ambiguity:
                    raise StaticAmbiguity(node, msg) from e
                else:
                    self._state.msg(msg, ctx=elt)

        try:
            return _AST_SEQUENCE_TO_TYPE[type(node)](values)
        except TypeError as e:
            raise StaticValueError(node, "invalid literal") from e

    visit_Tuple = visit_List
    visit_Set = visit_List

    def visit_BinOp(self, node: ast.BinOp, path: List[ast.AST]) -> LiteralValue:
        lval = ensure_literal(self.visit(node.left, path))
        rval = ensure_literal(self.visit(node.right, path))
        try:
            return op2func(node.op, type(node))(lval, rval)
        except Exception as e:
            raise StaticEvaluationError(
                node, f"{e.__class__.__name__} in binary operation: {e}"
            ) from e

    def visit_BoolOp(self, node: ast.BoolOp, path: List[ast.AST]) -> LiteralValue:
        for val_node in node.values:
            val = ensure_literal(self.visit(val_node, path))
            if (isinstance(node.op, ast.Or) and val) or (
                isinstance(node.op, ast.And) and not val
            ):
                return val
        return val

    def visit_UnaryOp(self, node: ast.UnaryOp, path: List[ast.AST]) -> LiteralValue:
        val = ensure_literal(self.visit(node.operand, path))
        try:
            ret = op2func(node.op, type(node))(val)
        except Exception as e:
            raise StaticEvaluationError(
                node, f"{e.__class__.__name__} in unary operator"
            ) from e
        return ret

    def visit_Compare(self, node: ast.Compare, path: List[ast.AST]) -> LiteralValue:
        """comparison operators, including chained comparisons (a<b<c)"""
        lval = ensure_literal(self.visit(node.left, path))
        results: List[LiteralValue] = []
        for oper, rnode in zip(node.ops, node.comparators):
            rval = ensure_literal(self.visit(rnode, path))
            try:
                ret = op2func(oper, type(node))(lval, rval)
            except Exception as e:
                raise StaticEvaluationError(
                    node, f"{e.__class__.__name__} in comparator"
                ) from e
            results.append(ret)
            lval = rval
        if len(results) == 1:
            return results[0]
        out: LiteralValue = True
        for ret in results:
            out = out and ret
        return out

    def visit_Subscript(self, node: ast.Subscript, path: List[ast.AST]) -> LiteralValue:
        if not isinstance(getattr(node, "ctx", ast.Load()), ast.Load):
            raise StaticTypeError(node.ctx, expected="Load")
        value = ensure_literal(self.visit(node.value, path))
        if not isinstance(value, (dict, list, tuple, str, bytes)):
            raise StaticTypeError(value, expected="subscriptable")
        slc = ensure_literal(self.visit(node.slice, path))
        try:
            return value[slc]  # type: ignore
        except Exception as e:
            raise StaticEvaluationError(node, str(e)) from e

    def visit_Slice(self, node: ast.Slice, path: List[ast.AST]) -> slice:
        lower = (
            ensure_literal(self.visit(node.lower, path))
            if node.lower is not None
            else None
        )
        upper = (
            ensure_literal(self.visit(node.upper, path))
            if node.upper is not None
            else None
        )
        step = (
            ensure_literal(self.visit(node.step, path))
            if node.step is not None
            else None
        )
        return slice(lower, upper, step)

    if sys.version_info < (3,8):
        def visit_NameConstant(self, node:ast.NameConstant, path:Any) -> LiteralValue:
            return self.visit_Constant(node, path)
        
        def visit_Ellipsis(self, node:ast.Ellipsis, path:Any) -> LiteralValue:
            return ...

        def visit_Num(self, node:ast.Num, path:Any) -> LiteralValue:
            v = node.n
            if not isinstance(v, numbers.Number):
                raise StaticTypeError(v, expected="literal")
            return v
        
        def visit_Str(self, node:Union[ast.Str, ast.Bytes], path:Any) -> LiteralValue:
            v = node.s
            if not isinstance(v, (str, bytes)):
                raise StaticTypeError(v, expected="literal")
            return v
        
        visit_Bytes = visit_Str

    if sys.version_info < (3,9):

        def visit_Index(self, node: ast.Index, path: List[ast.AST]) -> LiteralValue:
            return ensure_literal(self.visit(node.value, path))
    
    if sys.version_info >= (3,8):

        def visit_NamedExpr(self, node: ast.NamedExpr, path: List[ast.AST]) -> LiteralValue:
            return ensure_literal(self.visit(node.value, path))

