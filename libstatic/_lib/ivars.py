from __future__ import annotations

from _ast import Module
import ast
from typing import Any, Dict, List, Mapping, Sequence, TYPE_CHECKING

from .shared import LocalStmtVisitor, StmtVisitor

if TYPE_CHECKING:
    from .model import Def
    
def is_instance_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool: 
    args = node.args.args
    if (
        len(args) == 0
        # special methods known not to be instance methods,
        # TODO: are there more?
        or node.name in {"__new__", 
                            "__init_subclass__",
                            "__class_getitem__", 
                            "__subclasscheck__", 
                            "__subclasshook__", }
        or any((getattr(d, "id", None)
                in {"classmethod", "staticmethod"}
                for d in node.decorator_list))
        # we include only methods that follows the 
        # 'self' convention to exclude code like
        # class C:
        #   def stmeth(thing):
        #       thing.var = 2
        #   stmeth = staticmethod(stmeth)
        or not args[0].arg=='self'
    ):
        # not a typical instance method
        return False
    return True

class IVarsVisitor(LocalStmtVisitor):
    def __init__(self, chains:Mapping[ast.AST, Def]) -> None:
        self.chains = chains
        self.ivars: Dict[str, List[Def]] = {}

    # gather attributes of 'self'
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not is_instance_method(node):
            # not a typical instance method
            return

        args = node.args.args
        self_def = self.chains[args[0]]
        for use in (u for u in self_def.users() if isinstance(u.node, ast.Name)):
            try:
                attr = next(attr for attr in use.users() 
                            if isinstance(attr.node, ast.Attribute) 
                            and attr.node.value is use.node)
            except StopIteration:
                continue
            else:
                self.ivars.setdefault(attr.name(), []).append(attr)

    visit_AsyncFunctionDef = visit_FunctionDef

def _compute_ivars(chains: Mapping[ast.AST, Def], cls: ast.ClassDef) -> Mapping[str, Sequence[Def]]:
    visitor = IVarsVisitor(chains)
    visitor.generic_visit(cls)
    return visitor.ivars

class ComputeInstanceVariables(StmtVisitor):
    def __init__(self, chains:Mapping[ast.AST, Def], ) -> None:
        self.chains = chains
    def visit_Module(self, node: Module) -> Any:
        self._result: Dict[ast.ClassDef, Mapping[str, Sequence[Def]]] = {}
        self.generic_visit(node)
        return self._result
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._result[node] = _compute_ivars(self.chains, node)
        self.generic_visit(node)