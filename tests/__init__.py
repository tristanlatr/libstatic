import ast

from libstatic._lib.exceptions import NodeLocation

def location(node:ast.AST, filename:'str|None') -> str:
    return str(NodeLocation.make(node, filename=filename))
