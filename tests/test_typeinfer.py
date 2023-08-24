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
from typing import Any
from unittest import TestCase
import pytest

from libstatic import Project, StaticException
from libstatic._analyzer.typeinfer import _AnnotationToType
from libstatic._lib.shared import StmtVisitor, unparse

@pytest.mark.parametrize(
        ("source", "expected"), 
        [
        ("var: None", ["None",""]),
        ("import typing as t\nvar: t.Generic[t.T]",             ["Generic[T]","typing"]), 
        ("import typing as t\nvar: t.Generic[t.T, t._KV]",      ["Generic[T, _KV]","typing"]),
        ("import typing as t\nfrom typing import Generic\nvar: Generic[t.T]",       ["Generic[T]","typing"]),
        ("import typing as t\nvar: t.Tuple[t.T,...]",         ["Tuple[T, ...]","typing"]), 
        ("import typing as t\nfrom mod import _model as m\nfrom typing import Optional\nvar: m.TreeRoot[Optional[t.T]]",      ["TreeRoot[Optional[T]]","typing,mod._model"]),
        ("import typing as t\nfrom mod import _model as m\nfrom typing import Optional\nvar: 'm.TreeRoot[Optional[t.T]]'",        ["TreeRoot[Optional[T]]","typing,mod._model"]),
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

    annotation, _ = expected
    if annotation.startswith('Union'):
        assert type_annotation.is_union
    if annotation.startswith('Literal'):
        assert type_annotation.is_literal

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

    with pytest.raises(StaticException):
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

    # math operations
    ('3 + 2',       'int'),
    ('3 * 2',       'int'),
    ('3 + 2.',      'float'),
    ('3. + 2',      'float'),
    ('3 / 2',       'float'),
    ('"a" + "b"',   'str'),

    # binary "bool" operations
    ('3 and 2',     'int'),
    ('3 or 2',      'int'),
    ('3. and 2.',   'float'),
    ('3. or 2.',    'float'),

    # operations with known type
    ('not x',       'bool'),
    ('x is str',    'bool'),

    # operations with assumptions
    ('x in (1, 2, 3)',  'bool'),
    ('x < 10',          'bool'),
    ('~13',             'int'),
    ('+13',             'int'),

    # comprehensions
    ('[x for x in y]',      'list'),
    ('{x for x in y}',      'set'),
    ('{x: y for x in z}',   'dict'),
    ('(x for x in y)',      'Iterator'),

    # misc
    # ('Some(x)',             'Some'),
])
def test_expr(expr, type):
    mod = ast.parse(f'None\n{expr}')
    node = mod.body[-1].value
    p = Project()
    p.add_module(mod, 'test')
    p.analyze_project()
    t = p.state.get_type(node)
    assert t is not None
    assert t.annotation == type

class TestTypeInferStubs(TestCase):
    def test_expr_stubs(self):
        cases = [ 

        # collection constructors
        ('list()',      'list'),
        ('list(x)',     'list'),
        ('dict()',      'dict'),
        ('dict(x)',     'dict'),
        ('set()',       'set'),
        ('set(x)',      'set'),
        ('tuple()',     'tuple'),
        ('tuple(x)',    'tuple'),

        # other type constructors
        ('int()',       'int'),
        ('int(x)',      'int'),
        ('str()',       'str'),
        ('str(x)',      'str'),
        ('float()',     'float'),
        ('float(x)',    'float'),

        # builtin functions
        ('len(x)',          'int'),
        ('oct(20)',         'str'),

        ('import builtins as b; b.len(x)',          'int'),
        ('import builtins as b; b.oct(20)',         'str'),

        # methods of builtins
        ('"".join(x)',      'str'),
        ('[1,2].count(1)',  'int'),
        ('list(x).copy()',  'list[_T]'),
        ('[].copy()',       'list[_T]'),
        ('[].__iter__()',   'Iterator[_T]'),

        ('import math\nmath.sin(x)',  'float'),
        ('from math import sin\nsin(x)',       'float'),
        ('my_list = list\nmy_list(x)',   'list'),
        ('from datetime import *\ndate(1,2,3)',  'date'),
        # ('def g(x): return 0\ng(x)',         'int'),
        ('from pathlib import Path\nx:str|Path=...\nx.as_posix()', 'str'),
        ('from pathlib import Path,Patate\nx:str|Path|Patate=...\nx.as_posix()', 'str'),
        ('x = 13\nx',            'int'),
        ('import typing as t; x:t.Literal[13] = ...\nx.real',            'int'),
        ('import typing as t; x:t.Literal[None] = ...\nx',            'Literal[None]'),
        ('x = 1\nif x:\n  x=True\nx',            'int | bool'),
        
        ]

        p = Project(dependencies=1)
        expected = {}
        for i, (expr, type) in enumerate(cases):
            p.add_module(ast.parse(f'None\n{expr}'), f'test{i}')
            expected[f'test{i}'] = (expr, type)
        
        p.analyze_project()
        for (modname, (expr, type)) in expected.items():
            with self.subTest(expr):
                node = p.state.get_module(modname).node.body[-1].value
                t = p.state.get_type(node)
                assert t is not None, (expr, type)
                assert t.annotation == type, (expr, type)

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
    '[1][2]',
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
    p.analyze_project()
    t = p.state.get_type(node)
    assert t is None


@pytest.mark.parametrize('sig, type', [
    ('a: int', 'int'),
    ('b, a: int, c', 'int'),
    ('b: float, a: int, c: float', 'int'),
    ('*, a: int', 'int'),
    ('a: int, /', 'int'),
    ('a: list', 'list'),

    # *args and **kwargs
    ('*a: int', 'tuple[int, ...]'),
    ('*a: garbage', 'tuple'),
    ('*a', 'tuple'),
    ('**a: int', 'dict[str, int]'),
    ('**a: garbage', 'dict[str, Any]'),
    ('**a', 'dict[str, Any]'),

    # parametrized generics
    ('a: list[str]', 'list[str]'),
    ('a: list[garbage]', 'list'),
    ('a: dict[str, int]', 'dict[str, int]'),
    ('a: tuple[str, int, float]', 'tuple[str, int, float]'),
    ('a: tuple[str, garbage]', 'tuple'),

    # from default value
    ('a=2', 'int'),
])
def test_infer_type_from_signature(sig, type):
    given = f"""
        def f({sig}):
            return a
    """
    mod = ast.parse(dedent(given))
    node = mod.body[-1].body[-1]
    assert isinstance(node, ast.Return)
    
    p = Project()
    p.add_module(mod, 'test')
    p.analyze_project()
    t = p.state.get_type(node.value)

    assert t is not None
    assert t.annotation == type


@pytest.mark.parametrize('sig', [
    '',
    'b',
    'b: int',
    'a',
    'a: garbage',
    'a: garbage[int]',
    'a=None',
])
def test_cannot_infer_type_from_signature(sig):
    given = f"""
        def f({sig}):
            return a
    """
    mod = ast.parse(dedent(given))
    node = mod.body[-1].body[-1]
    assert isinstance(node, ast.Return)
    
    p = Project()
    p.add_module(mod, 'test')
    t = p.state.get_type(node.value)

    assert t is None


###

@pytest.mark.parametrize('src', [])
def test_reveal(src:str) -> None:
    node = ast.parse(dedent(src))
    p = Project()
    p.add_module(node, 'test')
    p.analyze_project()

    class RevealVisitor(StmtVisitor):
        def visit_Expr(self, node: ast.Expr) -> Any:
            v = node.value
            if not isinstance(v, ast.Call): 
                return
            f = v.func
            if not isinstance(f, ast.Name): 
                return
            if f.id == 'reveal_type':
                expr, expected = v.args
                assert isinstance(expected, ast.Constant)
                typ = p.state.get_type(expr)
                expected_value = getattr(expected, 'value')
                if expected_value is None:
                    assert typ is None
                else:
                    assert typ is not None, f'cannot infer {unparse(ast.Expr(expr))}'
                    assert typ.annotation == expected_value
    
    RevealVisitor().visit(node)

@pytest.mark.parametrize('src', [
    r'''
    v = 2
    reveal_type(v,'int')
    ''',

    r'''
    class C:
        v = 2
    reveal_type(C.v,'int')
    ''',

    r'''
    from typing import AnyStr
    v:AnyStr = b'bb'
    reveal_type(v,'AnyStr')
    ''',

    r'''
    from typing import Union
    def f(x:int, v, y:Union[list[int], tuple[int]]) -> None:
        reveal_type(v, None)
        reveal_type(x,'int')
        reveal_type(y,'list[int] | tuple[int]')
    ''',

    '''
    class C:
        ...
    reveal_type(C,'Type[C]')
    ''',

    '''
    def func_no_ann(x:int):
        ...
    def func_ann(x:int) -> None:
        ...
    reveal_type(func_no_ann,'Callable[Any, Any]')
    reveal_type(func_ann,'Callable[Any, None]')
    ''',

    '''
    v = None
    x: None
    reveal_type(v,'None')
    reveal_type(x,'None')
    ''',
])
def test_reveal_types(src:str) -> None:
    test_reveal(src)