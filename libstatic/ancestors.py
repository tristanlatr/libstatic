import ast
from typing import Dict, List


class Ancestors(ast.NodeVisitor):
    """
    Build the ancestor tree, that associates a node to the list of node visited
    from the root node (the Module) to the current node
    """

    def __init__(self) -> None:
        self.parents: Dict[ast.AST, List[ast.AST]] = {}
        self._current: List[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.parents[node] = list(self._current)
        self._current.append(node)
        super().generic_visit(node)
        self._current.pop()
