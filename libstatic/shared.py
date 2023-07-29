import ast
from typing import Optional, List, overload, Any, Union
import sys

if sys.version_info >= (3,9):
    from ast import unparse as to_source
else:
    from astor import to_source

def node2dottedname(node: Optional[ast.AST]) -> Optional[List[str]]:
    """
    Resove expression composed by L{ast.Attribute} and L{ast.Name} nodes to a list of names. 
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