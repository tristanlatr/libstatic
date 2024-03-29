from __future__ import annotations

import ast
from typing import Callable, Optional, Tuple, Union, TypeVar, cast

from .shared import node2dottedname

T = TypeVar(
    "T", bound=Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef]
)
_astT = TypeVar("_astT", bound=ast.AST)

def fix_ast_location(new_node, old_node):
    return ast.fix_missing_locations(ast.copy_location(new_node, old_node))


def is_assert_False(node):
    return (
        isinstance(node, ast.Assert)
        and isinstance(node.test, (ast.Constant, ast.NameConstant))
        and bool(node.test.value) is False
    )


def is_Yield(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(
        stmt.value, (ast.Yield, ast.YieldFrom)
    )


class Transform(ast.NodeTransformer):
    """
    Transform the ast such that the code is more easy to understand.

        - Removes dead code
        - Transform supported __all__ operations into regular assignments
    """

    # TODO: unstring annotations?

    def __init__(self) -> None:
        super().__init__()
        # stack of nodes
        self._node_stack = []  # type:list[ast.AST]
        self._dead_blocks = set()  # type: set[Tuple[ast.AST , str]]
        self._control_flow_jumps = []  # type:list[Callable[[ast.AST], bool]]

    def generic_visit(self, node:_astT) -> Optional[_astT]: # type: ignore[override]
        # simply remove dead code, as a side effect this might creates
        # function or classes with an empty body, which is not correct python code.
        if self.current_block(node) in self._dead_blocks:
            # prevent removing dead yield and yield from statements
            # because they make the function a generator.
            if not is_Yield(node):
                return None

        self._node_stack.append(node)
        super().generic_visit(node)
        self._node_stack.pop()
        return node

    def transform(self, node: T) -> T:
        assert not self._node_stack

        self._node_stack.append(node)
        if isinstance(node, (ast.Module, ast.ClassDef)):
            self._control_flow_jumps.append(
                lambda n: type(n).__name__ in {"Raise"} or is_assert_False(n)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # must be a function
            self._control_flow_jumps.append(
                lambda n: type(n).__name__ in {"Return", "Raise"} or is_assert_False(n)
            )
        else:
            raise RuntimeError(f"unexpected {type(node)}")

        # does not call self.generic_visit() since we don't want to
        # remove whole function or classes, also it raises an IndexError
        # when calling current_block() when visiting a root node.
        node = cast(T, super().generic_visit(node))
        self._control_flow_jumps.pop()
        self._node_stack.pop()
        assert not self._node_stack
        return node

    def current(self) -> ast.AST:
        return self._node_stack[-1]

    def current_block(self, node: ast.AST) -> Tuple[ast.AST, str]:
        # tell in wich block the given node lives in.
        # the given node must be part of the direct children
        # of the current node
        current = self.current()
        for fieldname, value in ast.iter_fields(current):
            if value is node or (isinstance(value, (list, tuple)) and node in value):
                break
        else:
            raise RuntimeError(f"node {node} not found in {current}")
        return current, fieldname

    def visit_Module(self, node:ast.Module) -> None:
        assert False

    def visit_scope(self, node:T) -> T:
        return self.__class__().transform(node)

    visit_AsyncFunctionDef = visit_FunctionDef = visit_ClassDef = visit_scope

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        tnode = self.generic_visit(node)
        if tnode is None:
            return None
        # if both the body and the else branch are dead ends,
        # then the rest of the block is dead as well.
        if all(b in self._dead_blocks for b in [(tnode, "body"), (tnode, "orelse")]):
            self._dead_blocks.add(self.current_block(tnode))
        return tnode

    def visit_loop(self, node:ast.AST) -> Optional[ast.AST]:
        self._control_flow_jumps.append(
            lambda n: type(n).__name__ in {"Continue", "Break"}
        )
        tnode = self.generic_visit(node)
        self._control_flow_jumps.pop()
        return tnode

    visit_For = visit_AsyncFor = visit_While = visit_loop

    def visit_jump(self, node: ast.AST) -> Optional[ast.AST]:
        # Union[ast.Break, ast.Continue, ast.Return, ast.Raise]
        tnode = self.generic_visit(node)
        if tnode is None:
            # already in a dead block
            return None

        for exits in reversed(self._control_flow_jumps):
            if exits(tnode):
                self._dead_blocks.add(self.current_block(tnode))
                break

        return tnode

    visit_Raise = (
        visit_Return
    ) = visit_Break = visit_Continue = visit_Assert = visit_jump

    def visit_AugAssign(self, node: ast.AugAssign) -> Optional[ast.AST]:
        tnode = self.generic_visit(node)
        if tnode is None:
            return None
        # Only transform top-level __all__ augmentd assignments into regular assignments
        if node2dottedname(tnode.target) == ["__all__"] and not any(
            isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for n in self._node_stack
        ):
            return fix_ast_location(
                ast.Assign(
                    targets=[ast.Name(id="__all__", ctx=ast.Store())],
                    value=ast.BinOp(
                        left=ast.Name(id="__all__", ctx=ast.Load()),
                        op=tnode.op,
                        right=tnode.value,
                    ),
                    type_comment=None,
                ),
                tnode,
            )
        else:
            return tnode

    # Transform statement level __all__.extend() and __all__.append()
    # into assignments
    def visit_Expr(self, node:ast.Expr) -> Optional[ast.AST]:
        tnode = self.generic_visit(node)
        if tnode is None:
            return None

        if any(
            isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for n in self._node_stack
        ):
            return tnode

        v = tnode.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)):
            return tnode

        o = v.func.value
        a = v.func.attr

        if len(v.args) != 1 or len(v.keywords) != 0:
            return tnode

        # We can safely apply this transformation because
        # we know __all__ should be a list or tuple.
        if node2dottedname(o) == ["__all__"]:
            aug = None
            if a == "extend":
                # 'o' must be a list, but maybe v.args[0] is just an iterable
                # we use ast.Starred instead.
                aug = ast.Assign(
                    targets=[ast.Name("__all__", ast.Store())],
                    value=ast.List(
                        elts=[
                            ast.Starred(ast.Name("__all__", ast.Load()), ast.Load()),
                            ast.Starred(v.args[0], ast.Load()),
                        ], ctx=ast.Load()
                    ),
                    type_comment=None,
                )
            elif a == "append":
                aug = ast.Assign(
                    targets=[ast.Name("__all__", ast.Store())],
                    value=ast.List(
                        elts=[
                            ast.Starred(ast.Name("__all__", ast.Load()), ast.Load()),
                            v.args[0],
                        ], ctx=ast.Load()
                    ),
                    type_comment=None,
                )
            if aug:
                return fix_ast_location(aug, tnode)

        return tnode
