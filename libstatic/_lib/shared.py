import ast
from typing import Optional, List, overload, Any, Union, TYPE_CHECKING
import sys

if sys.version_info >= (3,9):
    from ast import unparse as to_source
else:
    from astor import to_source

def node2dottedname(node: Optional[ast.AST]) -> Optional[List[str]]:
    """
    Resove expression composed by `ast.Attribute` and `ast.Name` nodes to a list of names. 
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    parts.reverse()
    return parts

@overload
def ast_node_name(n:Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Name, ast.arg, ast.alias]) -> 'str': # type: ignore[misc]
    ...
@overload
def ast_node_name(n:Any) -> None:
    ...
def ast_node_name(n:Any) -> 'str|None':
    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return n.name
    elif isinstance(n, ast.Name):
        return n.id
    elif isinstance(n, ast.arg):
        return n.arg
    elif isinstance(n, ast.alias):
        base = n.name.split(".", 1)[0]
        return n.asname or base
    return None

def unparse(node:ast.AST) -> str:
    try:
        return to_source(node)
    except Exception:
        return '??'

class StmtVisitor(ast.NodeVisitor):
    """
    Does not recurse on leaf type statements' content by default.
    """

    def visit_stmt(self, node:ast.stmt) -> None:
        pass
    
    if not TYPE_CHECKING:
        # let's just hide this from mypy
        visit_Assign = visit_AugAssign = visit_AnnAssign = visit_Expr = visit_stmt
        visit_Return = visit_Print = visit_Raise = visit_Assert = visit_stmt
        visit_Pass = visit_Break = visit_Continue = visit_Delete = visit_stmt
        visit_Global = visit_Nonlocal = visit_Exec = visit_stmt
        visit_Import = visit_ImportFrom = visit_stmt

class LocalStmtVisitor(ast.NodeVisitor):
    """
    Like `StmtVisitor` but does not recurse on functions or classes by default.
    """

    if not TYPE_CHECKING:
        visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = StmtVisitor.visit_stmt