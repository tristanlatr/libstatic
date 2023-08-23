# from https://github.com/orsinium-labs/astypes/blob/0.2.6/tests/test_handlers.py
# MIT License

#  2022 Gram

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ast
from textwrap import dedent
import pytest

from libstatic import Project
from libstatic._analyzer.typeinfer import _AnnotationToType

@pytest.mark.parametrize(
        ("source", "expected"), 
        [
        ("var: None", ["None",""]),
        ("var: typing.Generic[T]", ["Generic[T]","typing"]), 
        ("var: typing.Generic[T, _KV]", ["Generic[T, _KV]","typing"]),
        ("from typing import Generic\nvar: Generic[T]", ["Generic[T]","typing"]),
        ("var: typing.Tuple[T,...]", ["Tuple[T, ...]","typing"]), 
        ("from mod import _model as m\nfrom typing import Optional\nvar: m.TreeRoot[Optional[T]]", ["TreeRoot[Optional[T]]","typing,mod._model"]),
        ("from mod import _model as m\nfrom typing import Optional\nvar: 'm.TreeRoot[Optional[T]]'", ["TreeRoot[Optional[T]]","typing,mod._model"]),
        ("var: dict[str, str]", ["dict[str, str]",'']),
        ("var: dict[str, str] | dict[str, int]", ["dict[str, str] | dict[str, int]",'']),
        ("import typing as t\nvar: t.Union[dict[str, str], dict[str, int]]", ["dict[str, str] | dict[str, int]",'typing']),
        ("import typing as t\nvar: t.Literal[True, False]", ["Literal[True, False]",'typing']),
        ("import typing as t\nvar: t.Literal['string']", ["Literal['string']",'typing']),
        ("import typing as t\nvar: dict[t.Type, t.Callable[[t.Any], t.Type]]", ["dict[Type, Callable[Any, Type]]",'typing'])
        ]
    )
def test_annotation_to_type(source:str, expected:str) -> None:
    mod = ast.parse(source)
    p = Project()
    m = p.add_module(mod, 'test')
    p.analyze_project()

    annotation2type = lambda expr: _AnnotationToType(p.state, m).visit(expr)

    type_annotation = annotation2type(mod.body[-1].annotation)

    # type_annotation_method2 = get_type(mod.symbols['var'][-1])
    # assert type_annotation == type_annotation_method2

    assert not type_annotation.unknown
    imports = '\n'.join(type_annotation.imports)
    annotation, imports_contains = expected
    if annotation.startswith('Union'):
        assert type_annotation.is_union
    if annotation.startswith('Literal'):
        assert type_annotation.is_literal
    for i in imports_contains.split(','):
        assert i in imports, f"{i!r} not in {imports}"
    assert type_annotation.annotation == annotation

    # smoke test
    ast.parse(type_annotation.annotation)

@pytest.mark.parametrize(
        ("source"), 
        [
            ("var: False"),
            ("var: True"),
            ("var: 'Syntax error'"),
            ("var: 1 + 2"),
            ("var: thing[1]"), 
            # ("var: [1,2,3]"), 
        ])
def test_annotation_to_error(source:str) -> None:
    mod = ast.parse(source)
    p = Project()
    m = p.add_module(mod, 'test')
    p.analyze_project()

    annotation2type = lambda expr: _AnnotationToType(p.state, m).visit(expr)

    with pytest.raises(ValueError):
        annotation2type(mod.body[-1].annotation)

@pytest.mark.parametrize('expr, type', [
    # literals
    ('1',       'int'),
    ('1.2',     'float'),
    ('"hi"',    'str'),
    ('f"hi"',   'str'),
    ('b"hi"',   'bytes'),
    ('""',      'str'),
    ('None',    'None'),
    ('True',    'bool'),

    # collection literals
    ('[]',          'list'),
    ('[1]',         'list[int]'),
    ('[1, 2]',      'list[int]'),
    ('[x]',         'list'),
    ('[1, ""]',     'list[int | str]'),
    ('()',          'tuple'),
    ('(1,)',        'tuple[int]'),
    ('(x,)',        'tuple'),
    ('(1, x)',      'tuple'),
    ('(1, 2)',      'tuple[int, int]'),
    ('(1, "")',     'tuple[int, str]'),
    ('{}',          'dict'),
    ('{1:2}',       'dict[int, int]'),
    ('{1:x}',       'dict[int, Any]'),
    ('{1:x,2:y}',   'dict[int, Any]'),
    ('{1:x,"":y}',  'dict[int | str, Any]'),
    ('{x:1,y:2}',   'dict[Any, int]'),
    ('{x:1,y:""}',  'dict[Any, int | str]'),
    ('{1,2}',       'set[int]'),
    ('{1,2}',       'set[int]'),
    ('{1,x}',       'set'),
    ('{x}',         'set'),
    ('{x,y}',       'set'),
    ('{1,""}',      'set[int | str]'),

    # # collection constructors
    # ('list()',      'list'),
    # ('list(x)',     'list'),
    # ('dict()',      'dict'),
    # ('dict(x)',     'dict'),
    # ('set()',       'set'),
    # ('set(x)',      'set'),
    # ('tuple()',     'tuple'),
    # ('tuple(x)',    'tuple'),

    # # other type constructors
    # ('int()',       'int'),
    # ('int(x)',      'int'),
    # ('str()',       'str'),
    # ('str(x)',      'str'),
    # ('float()',     'float'),
    # ('float(x)',    'float'),

    # # math operations
    # ('3 + 2',       'int'),
    # ('3 * 2',       'int'),
    # ('3 + 2.',      'float'),
    # ('3. + 2',      'float'),
    # ('3 / 2',       'float'),
    # ('"a" + "b"',   'str'),

    # # binary "bool" operations
    # ('3 and 2',     'int'),
    # ('3 or 2',      'int'),
    # ('3. and 2.',   'float'),
    # ('3. or 2.',    'float'),

    # # operations with known type
    # ('not x',       'bool'),
    # ('x is str',    'bool'),

    # # operations with assumptions
    # ('x in (1, 2, 3)',  'bool'),
    # ('x < 10',          'bool'),
    # ('~13',             'int'),
    # ('+13',             'int'),

    # # methods of builtins
    # ('"".join(x)',      'str'),
    # ('[1,2].count(1)',  'int'),
    # ('list(x).copy()',  'list'),
    # ('[].copy()',       'list'),
    # ('[].__iter__()',   'Iterator'),

    # # builtin functions
    # ('len(x)',          'int'),
    # ('oct(20)',         'str'),

    # # comprehensions
    # ('[x for x in y]',      'list'),
    # ('{x for x in y}',      'set'),
    # ('{x: y for x in z}',   'dict'),
    # ('(x for x in y)',      'Iterator'),

    # misc
    # ('Some(x)',             'Some'),
])
def test_expr(expr, type):
    mod = ast.parse(f'None\n{expr}')
    node = mod.body[-1].value
    p = Project()
    p.add_module(mod, 'test')
    t = p.state.get_type(node)
    assert t is not None
    assert t.annotation == type


@pytest.mark.parametrize('expr', [
    'min(x)',
    'x',
    'X',
    'WAT',
    'wat()',
    'WAT()',
    '+x',
    'x + y',
    '1 + y',
    'x + 1',
    '"a" + 1',
    'str.wat',
    '"hi".wat',
    'None.hi',
    'None.hi()',
    '"hi".wat()',
    'wat.wat',
    'super().something()',
    'len(x).something()',
    '[].__getitem__(x)',
    'x or y',
    'x and y',
    'x = None; x = b(); x',
    'def g() -> x: pass\ng()',
])
def test_cannot_infer_expr(expr):
    mod = ast.parse(expr)
    node = mod.body[-1].value
    p = Project()
    p.add_module(mod, 'test')
    t = p.state.get_type(node)
    assert t is None

# @pytest.mark.parametrize('setup, expr, type', [
#     ('import math',                 'math.sin(x)',  'float'),
#     ('from math import sin',        'sin(x)',       'float'),
#     ('my_list = list',              'my_list(x)',   'list'),
#     ('def g(x): return 0',          'g(x)',         'int'),
#     ('x = 13',                      'x',            'int'),
#     ('x = 1\nif x:\n  x=True',      'x',            'int | bool'),
#     ('from datetime import *',      'date(1,2,3)',  'date'),
# ])
# def test_more_inference(setup, expr, type):
#     mod = ast.parse(f'{setup}\n{expr}')
#     node = mod.body[-1].value
#     p = Project(dependencies=True)
#     p.add_module(mod, 'test')
#     t = p.state.get_type(node)

#     assert t is not None
#     assert t.annotation == type


# @pytest.mark.parametrize('sig, type', [
#     ('a: int', 'int'),
#     ('b, a: int, c', 'int'),
#     ('b: float, a: int, c: float', 'int'),
#     ('*, a: int', 'int'),
#     ('a: int, /', 'int'),
#     ('a: list', 'list'),

#     # *args and **kwargs
#     ('*a: int', 'tuple[int]'),
#     ('*a: garbage', 'tuple'),
#     ('*a', 'tuple'),
#     ('**a: int', 'dict[str, int]'),
#     ('**a: garbage', 'dict[str, Any]'),
#     ('**a', 'dict[str, Any]'),

#     # parametrized generics
#     ('a: list[str]', 'list[str]'),
#     ('a: list[garbage]', 'list'),
#     ('a: dict[str, int]', 'dict[str, int]'),
#     ('a: tuple[str, int, float]', 'tuple[str, int, float]'),
#     ('a: tuple[str, garbage]', 'tuple'),
# ])
# def test_infer_type_from_signature(sig, type):
#     given = f"""
#         def f({sig}):
#             return a
#     """
#     mod = ast.parse(dedent(given))
#     node = mod.body[-1].body[-1]
#     assert isinstance(node, ast.Return)
    
#     p = Project()
#     p.add_module(mod, 'test')
#     t = p.state.get_type(node.value)

#     assert t is not None
#     assert t.annotation == type


# @pytest.mark.parametrize('sig', [
#     '',
#     'b',
#     'b: int',
#     'a',
#     'a: garbage',
#     'a: garbage[int]',
# ])
# def test_cannot_infer_type_from_signature(sig):
#     given = f"""
#         def f({sig}):
#             return a
#     """
#     mod = ast.parse(dedent(given))
#     node = mod.body[-1].body[-1]
#     assert isinstance(node, ast.Return)
    
#     p = Project()
#     p.add_module(mod, 'test')
#     t = p.state.get_type(node.value)

#     assert t is None
