import ast
import enum
from functools import lru_cache
import inspect
import numbers
import sys

from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, TypeAlias, Union

# Credits to pydoctor project for most of these utilities
# https://github.com/twisted/pydoctor/blob/master/pydoctor/astutils.py

_AssingT = Union[ast.Assign, ast.AnnAssign, ast.AugAssign]

def iterassignfull(node:_AssingT) -> Iterator[Tuple[Optional[List[str]], ast.expr]]:
    """
    Utility function to iterate assignments targets. 
    Like L{iterassign} but returns the C{ast.expr} of the target as well.
    """
    for target in node.targets if node.__class__.__name__=='Assign' else [node.target]:
        yield node2dottedname(target) , target

def node2dottedname(node: Optional[ast.AST]) -> Optional[List[str]]:
    """
    Resove expression composed by L{ast.Attribute} and L{ast.Name} nodes to a list of names. 
    """
    parts = []
    while node.__class__.__name__ == 'Attribute':
        parts.append(node.attr)
        node = node.value
    if node.__class__.__name__ == 'Name':
        parts.append(node.id)
    else:
        return None
    parts.reverse()
    return parts

class AnnotationStringParser(ast.NodeTransformer):
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

def ast_equals(node1:Any, node2:Any) -> bool:
    """
    Compare two AST nodes and tell if they represent the same code.
    """
    if not isinstance(node2, type(node1)):
        return False
    if isinstance(node1, ast.AST):
        try:
            for original_field, other_field in zip(
                ast.iter_fields(node1), ast.iter_fields(node2), strict=True
            ):
                _, original_value = original_field
                __, other_value = other_field
                if not ast_equals(original_value, other_value):
                    return False
            else:
                return True
        except ValueError:
            return False
    elif isinstance(node1, (list, tuple)):
        try:
            for original_ele, other_ele in zip(
                node1, node2, strict=True
            ):
                if not ast_equals(original_ele, other_ele):
                    return False
            else:
                return True
        except ValueError:
            return False
    else:
        return bool(node1 == node2)

def except_handler_types(h:ast.ExceptHandler) -> Optional[Sequence[Optional[str]]]:
    """
    Extract the types this exception handler catches.
    """
    dottedname = lambda n: '.'.join(node2dottedname(n) or ('',)) or None
    
    if isinstance(h.type, ast.Tuple):
        return [dottedname(n) for n in h.type.elts]
    elif h.type is None:
        return None
    else:
        return [dottedname(h.type)]


class VariableArgument(str):
    """
    Encapsulate the name of C{vararg} parameters in C{annotations} mapping keys as returned by L{annotations_from_function}.
    """

class KeywordArgument(str):
    """
    Encapsulate the name of C{kwarg} parameters in C{annotations} mapping keys as returned by L{annotations_from_function}.
    """

def annotations_from_function(
        func: Union[ast.AsyncFunctionDef, ast.FunctionDef]
        ) -> Mapping[str, Optional[ast.expr]]:
    """Get annotations from a function definition.
    @param func: The function definition's AST.
    @return: Mapping from argument name to annotation.
        The name C{return} is used for the return type.
        Unannotated arguments are omitted.
    """

    def _get_all_args() -> Iterator[ast.arg]:
        base_args = func.args
        # New on Python 3.8 -- handle absence gracefully
        try:
            yield from base_args.posonlyargs
        except AttributeError:
            pass
        yield from base_args.args
        varargs = base_args.vararg
        if varargs:
            varargs.arg = VariableArgument(varargs.arg)
            yield varargs
        yield from base_args.kwonlyargs
        kwargs = base_args.kwarg
        if kwargs:
            kwargs.arg = KeywordArgument(kwargs.arg)
            yield kwargs
    def _get_all_ast_annotations() -> Iterator[Tuple[str, Optional[ast.expr]]]:
        for arg in _get_all_args():
            yield arg.arg, arg.annotation
        returns = func.returns
        if returns:
            yield 'return', returns
    return {
        # Include parameter names even if they're not annotated, so that
        # we can use the key set to know which parameters exist and warn
        # when non-existing parameters are documented.
        name: None if value is None else value
        for name, value in _get_all_ast_annotations()
        }

def parameters_from_function(func: Union[ast.AsyncFunctionDef, ast.FunctionDef]
    ) -> Sequence[inspect.Parameter]:
    """
    Does not support:
    - parameter defaults.
    - posisiton only or keyword only parameters
        (they are considered as positional or keyword)
    """
    annotations = annotations_from_function(func)
    params = []
    for name, ann in annotations.items():
        if name == 'return':
            continue
        kind:inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if isinstance(name, VariableArgument):
            kind = inspect.Parameter.VAR_POSITIONAL
        elif isinstance(name, KeywordArgument):
            kind = inspect.Parameter.VAR_KEYWORD
        params.append(inspect.Parameter(str(name), kind=kind, 
                    default=inspect.Parameter.empty, 
                    annotation=ann))
    return params

NAMECONSTANT_TYPE = ast.Constant if sys.version_info >= (3,3) else ast.NameConstant

# Credits to scalpel project
# https://github.com/SMAT-Lab/Scalpel
def invert_expr(node:ast.expr) -> ast.expr:
    """
    Invert the operation in an ast node object (get its negation).
    Args:
        node: An ast node object.
    Returns:
        An ast node object containing the inverse (negation) of the input node.
    """
    inverse:Mapping[Type[ast.AST], Type[ast.AST]] = {ast.Eq: ast.NotEq,
               ast.NotEq: ast.Eq,
               ast.Lt: ast.GtE,
               ast.LtE: ast.Gt,
               ast.Gt: ast.LtE,
               ast.GtE: ast.Lt,
               ast.Is: ast.IsNot,
               ast.IsNot: ast.Is,
               ast.In: ast.NotIn,
               ast.NotIn: ast.In}

    if isinstance(node, ast.Compare):
        op:Type[ast.AST] = type(node.ops[0])
        inverse_node:ast.expr = ast.Compare(left=node.left, ops=[inverse[op]()],
                                   comparators=node.comparators)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, tuple(inverse)):
        op = type(node.op)
        inverse_node = ast.BinOp(node.left, inverse[op](), node.right)
    elif isinstance(node, NAMECONSTANT_TYPE):
        inverse_node = NAMECONSTANT_TYPE(value=not node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inverse_node = node.operand
    else:
        inverse_node = ast.UnaryOp(op=ast.Not(), operand=node)
    return inverse_node

class Context(enum.Enum):
    Load = 1
    Store = 2
    Del = 3

_CONTEXT_MAP = {
    'Load': Context.Load,
    'Store': Context.Store,
    'Del': Context.Del,
    'Param': Context.Store,
}

_AssignmentTargetExpr:TypeAlias = 'Union[ast.Attribute, ast.List, ast.Name, ast.Subscript, ast.Starred, ast.Tuple]'

@lru_cache()
def get_context(node: _AssignmentTargetExpr) -> Context:
    """
    Wraps the context ast context classes into a more friendly enumeration.
    Dynamically created nodes do not have the ctx field, in this case fall back to Load context.
    """

    # Just in case, we use getattr because dynamically created nodes do not have the ctx field.
    try:
        return _CONTEXT_MAP[type(getattr(node, 'ctx', ast.Load())).__name__] # type:ignore[index]
    except KeyError as e:
        raise ValueError(f"Can't get the context of {node!r}") from e

# Safe versions of functions to prevent denial of service issues
MAX_EXPONENT = 10000
MAX_STR_LEN = 2 << 17  # 256KiB
MAX_SHIFT = 1000

def safe_pow(base, exp):
    """safe version of pow"""
    if isinstance(exp, numbers.Number):
        if exp > MAX_EXPONENT:
            raise RuntimeError(f"Invalid exponent, max exponent is {MAX_EXPONENT}")
    return base ** exp


def safe_mult(arg1, arg2):
    """safe version of multiply"""
    if isinstance(arg1, str) and isinstance(arg2, int) and len(arg1) * arg2 > MAX_STR_LEN:
        raise RuntimeError(f"String length exceeded, max string length is {MAX_STR_LEN}")
    return arg1 * arg2


def safe_add(arg1, arg2):
    """safe version of add"""
    if isinstance(arg1, str) and isinstance(arg2, str) and len(arg1) + len(arg2) > MAX_STR_LEN:
        raise RuntimeError(f"String length exceeded, max string length is {MAX_STR_LEN}")
    return arg1 + arg2


def safe_lshift(arg1, arg2):
    """safe version of lshift"""
    if isinstance(arg2, numbers.Number):
        if arg2 > MAX_SHIFT:
            raise RuntimeError(f"Invalid left shift, max left shift is {MAX_SHIFT}")
    return arg1 << arg2

_OPERATORS = {'Is': lambda a, b: a is b,
             'IsNot': lambda a, b: a is not b,
             'In': lambda a, b: a in b,
             'NotIn': lambda a, b: a not in b,
             'Add': safe_add,
             'BitAnd': lambda a, b: a & b,
             'BitOr': lambda a, b: a | b,
             'BitXor': lambda a, b: a ^ b,
             'Div': lambda a, b: a / b,
             'FloorDiv': lambda a, b: a // b,
             'LShift': safe_lshift,
             'RShift': lambda a, b: a >> b,
             'Mult': safe_mult,
             'Pow': safe_pow,
             'MatMult': lambda a, b: a @ b,
             'Sub': lambda a, b: a - b,
             'Mod': lambda a, b: a % b,
             'And': lambda a, b: a and b,
             'Or': lambda a, b: a or b,
             'Eq': lambda a, b: a == b,
             'Gt': lambda a, b: a > b,
             'GtE': lambda a, b: a >= b,
             'Lt': lambda a, b: a < b,
             'LtE': lambda a, b: a <= b,
             'NotEq': lambda a, b: a != b,
             'Invert': lambda a: ~a,
             'Not': lambda a: not a,
             'UAdd': lambda a: +a,
             'USub': lambda a: -a}

def op2func(oper) -> Callable[[Any, Any], Any]:
    """Return function for operator nodes."""
    return _OPERATORS[oper.__class__.__name__]

def ast_node_name(n:ast.AST, modname:str='') -> 'str|None':
    if isinstance(n, (ast.ClassDef,
                                  ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
        return n.name
    elif isinstance(n, ast.Name):
        return n.id
    elif isinstance(n, ast.alias):
        base = n.name.split(".", 1)[0]
        return n.asname or base
    elif isinstance(n, ast.Module) and modname:
        return modname
    return None

def ast_repr(n:ast.AST, modname:str='') -> str:

    lineno = getattr(n,'lineno','?')
    col_offset = getattr(n,'col_offset','?')
    if isinstance(n, ast.expr):
        return f"{ast.dump(n)} at {modname or '<unknown>'}:{lineno}:{col_offset}"
    else:
        node_name = ast_node_name(n, modname)
        name = f':{node_name}' if node_name else ''
        return f"{n.__class__.__name__}{name} at {modname or '<unknown>'}:{lineno}:{col_offset}"
    