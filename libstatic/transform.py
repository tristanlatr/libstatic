import ast
from typing import Callable, Tuple

from libstatic.astutils import node2dottedname

def dottedname(expr:ast.expr) -> str:
    names = node2dottedname(expr)
    if not names:
        raise ValueError()
    return '.'.join(names)

def fix_ast_location(new_node, old_node):
    return ast.fix_missing_locations(ast.copy_location(new_node, old_node))

def is_assert_False(node):
    return (
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Constant)
        and bool(node.test.value) is False
    )

def is_Yield(stmt):
    return (isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)))

class Transform(ast.NodeTransformer):
    """
    Transform the ast such that the code is more easy
    to understand for static analyzers.

    - Removes dead code
    - Transform augmented assignments into regular ones
    - Transform supported __all__ operations into assignments
    - Support 'assert False' statements as a control flow jump
    """
    # TODO: unstring annotations

    def __init__(self):
        super().__init__()
        # stack of nodes
        self._node_stack = []  # type:list[ast.AST]
        self._dead_blocks = set()  # type: set[Tuple[ast.AST , str]]
        self._control_flow_jumps = []  # type:list[Callable[[ast.AST], bool]]

    def generic_visit(self, node):
        # simply remove dead code, as a side effect this might creates
        # function or classes with an empty body, which is not correct python code.
        if self.current_block(node) in self._dead_blocks:
            # prevent removing dead yield and yield from statements
            # because they make the function a generator.
            if not is_Yield(node):
                return None

        self._node_stack.append(node)
        node = super().generic_visit(node)
        self._node_stack.pop()
        return node

    def transform(self, node):
        # type: (ast.stmt) -> ast.stmt
        assert not self._node_stack

        self._node_stack.append(node)
        if isinstance(node, (ast.Module, ast.ClassDef)):
            self._control_flow_jumps.append(
                lambda n: type(n).__name__ in {"Raise"} or is_assert_False(n))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # must be a function
            self._control_flow_jumps.append(
                lambda n: type(n).__name__ in {"Return", "Raise"} or is_assert_False(n)
            )
        else:
            raise ValueError()

        # does not call self.generic_visit() since we don't want to
        # remove whole function or classes, also it raises an IndexError
        # when calling current_block() when visiting a root node.
        node = super().generic_visit(node)
        self._control_flow_jumps.pop()
        self._node_stack.pop()
        assert not self._node_stack

        return node

    def current(self):
        return self._node_stack[-1]

    def current_block(self, node:ast.AST) -> Tuple[ast.AST, str]:
        # tell in wich block the given node lives in.
        # the given node must be part of the direct children
        # of the current node
        current = self.current()
        for fieldname, value in ast.iter_fields(current):
            if value is node or (isinstance(value, (list, tuple)) and node in value):
                break
        else:
            raise RuntimeError(f'node {node} not found in {current}')
        return current, fieldname

    def visit_Module(self, node):
        assert False

    def visit_Classdef(self, node):
        return self.__class__().transform(node)

    visit_AsyncFunctionDef = visit_FunctionDef = visit_Classdef

    # blind spot: we need to account whether the names 
    # used in the if test are not redefined within the 
    # the body or the orelse clause of the if branch.
    def visit_If(self, node:ast.stmt):
        node = self.generic_visit(node)
        if node is None:
            return None
        # if both the body and the else branch are dead ends,
        # then the rest of the block is dead as well.
        if all(b in self._dead_blocks for b in [(node, "body"), (node, "orelse")]):
            self._dead_blocks.add(self.current_block(node))
        return node

    def visit_loop(self, node):
        self._control_flow_jumps.append(
            lambda n: type(n).__name__ in {"Continue", "Break"}
        )
        node = self.generic_visit(node)
        self._control_flow_jumps.pop()
        return node

    visit_For = visit_AsyncFor = visit_While = visit_loop

    def visit_jump(self, node:ast.stmt):
        # Union[ast.Break, ast.Continue, ast.Return, ast.Raise]
        node = self.generic_visit(node)
        if node is None:
            # already in a dead block
            return None
        
        for exits in reversed(self._control_flow_jumps):
            if exits(node):
                self._dead_blocks.add(self.current_block(node))
                break

        return node

    visit_Raise = visit_Return = visit_Break = visit_Continue = visit_Assert = visit_jump

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.Assign:
        node = self.generic_visit(node)
        if node is None:
            return None

        return fix_ast_location(
            ast.Assign(
                targets=[node.target],
                value=ast.BinOp(left=node.target, op=node.op, right=node.value),
                type_comment=None,
            ),
            node,
        )

    # Transform statement level __all__.extend() and __all__.append()
    # into assignments
    def visit_Expr(self, node):
        node = self.generic_visit(node)
        if node is None:
            return None

        if any(
            isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for n in self._node_stack
        ):
            return node

        v = node.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)):
            return node

        o = v.func.value
        a = v.func.attr

        if len(v.args) != 1 or len(v.keywords) != 0:
            return node

        # We can safely apply this transformation because
        # we know __all__ should be a list or tuple.
        if isinstance(o, ast.Name) and o.id == "__all__":
            aug = None
            if a == "extend":
                aug = ast.Assign(
                    targets=[
                        ast.Name(
                            o.id, ast.Store(), annotation=None, type_comment=None)
                    ],
                    value=ast.BinOp(
                        left=ast.Name(
                            o.id, ast.Load(), annotation=None, type_comment=None
                        ),
                        op=ast.Add(),
                        right=v.args[0],
                    ),
                    type_comment=None,
                )
            elif a == "append":
                aug = ast.Assign(
                    targets=[
                        ast.Name(
                            o.id, ast.Store(), annotation=None, type_comment=None)
                    ],
                    value=ast.BinOp(
                        left=ast.Name(
                            o.id, ast.Load(), annotation=None, type_comment=None
                        ),
                        op=ast.Add(),
                        right=ast.List(elts=[v.args[0]]),
                    ),
                    type_comment=None,
                )
            if aug:
                return fix_ast_location(aug, node)

        return node
