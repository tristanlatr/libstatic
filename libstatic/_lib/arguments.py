from __future__ import annotations

import ast
import inspect
from typing import Iterator
import dataclasses

@dataclasses.dataclass(frozen=True)
class ArgSpec:
    node: ast.arg
    kind: inspect._ParameterKind
    default: ast.expr | None = None


def iter_arguments(args: ast.arguments) -> Iterator[ArgSpec]:
    """
    Yields all arguments of the given `ast.arguments` node as `ArgSpec` instances.

    >>> node = ast.parse('def f(a:int, b:object=None, *, key:Callable, **kwargs):...')
    >>> args = iter_arguments(node.body[0].args)
    >>> parameters = [inspect.Parameter(a.node.arg, a.kind, 
    ... default=a.default or inspect.Parameter.empty, 
    ... annotation=a.node.annotation or inspect.Parameter.empty) for a in args]
    >>> sig = inspect.Signature(parameters)
    >>> str(sig)
    '(a:...Name..., b:...Name...=...Constant..., *, key:...Name..., **kwargs)'
    """
    posonlyargs = getattr(args, "posonlyargs", ())

    num_pos_args = len(posonlyargs) + len(args.args)
    defaults = args.defaults
    default_offset = num_pos_args - len(defaults)

    def get_default(index: int) -> ast.expr | None:
        assert 0 <= index < num_pos_args, index
        index -= default_offset
        return None if index < 0 else defaults[index]

    for i, arg in enumerate(posonlyargs):
        yield ArgSpec(arg, inspect.Parameter.POSITIONAL_ONLY, default=get_default(i))
    for i, arg in enumerate(args.args, start=len(posonlyargs)):
        yield ArgSpec(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=get_default(i))
    if args.vararg:
        yield ArgSpec(args.vararg, inspect.Parameter.VAR_POSITIONAL)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        yield ArgSpec(arg, inspect.Parameter.KEYWORD_ONLY, default=default)
    if args.kwarg:
        yield ArgSpec(args.kwarg, inspect.Parameter.VAR_KEYWORD)