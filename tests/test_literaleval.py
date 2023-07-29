import ast
from unittest import TestCase

from libstatic.model import Project
from libstatic.asteval import _LiteralEval, _ASTEval
from libstatic.exceptions import StaticUnknownValue, StaticException


class TestLiteralEval(TestCase):
    # not covered: module variable access with attribute
    def test_literal_eval(self):
        cases = [('v = 1.25', 1.25),
                 ('v = 3+4', 7), 
                 ('v = (x:=3+4)', 7), 
                 ('v = []', []),
                 ('v = [1,2,3]+["4",]', [1,2,3,"4"]),
                 ('x = [1,2,3]; v = [1,2,3, *x]', [1,2,3,1,2,3]),
                 ('v = ()', ()),
                 ('v = (1,2,3)+("4",)', (1,2,3,"4")),
                 ('v = []', []),
                 ('v = {1,2,"3"}', {1,2,"3"}),
                 ('class C: v=1\nv = C.v', 1),
                 ('class C:\n class V:\n  a =1\nv = C.V.a', 1),
                 ('x =1;v = x', 1),
                 ('v = (3,7,12)[:2]', (3,7)),
                 ('v = (3,7,12)[1:2]', (7,)),
                 ('v = (3,7,12)[:2]>=(3,0)', True),
                 ('v = True and False', False),
                 ('v = False or False', False),
                 ('v = False or True or False', True),
                 ('v = True and True and False', False),
                 ('v = 0 or 4', 4),
                 ('v = 0<10<=10<11', True),
                 ('v = 0<10<=10>11', False),
                 ('v = [1,2,3][1]', 2),
                 ('v = [1,2,3][-1]', 3),
                 ('v = +1+1-(+1-(-1))', 0),
                 ('v = ...', ...)
                 ]
        for code, expected in cases:
            with self.subTest(code):
                proj = Project()
                proj.add_module(ast.parse(code), 'm')
                proj.analyze_project()
                v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
                computed_value = proj.state.literal_eval(v.node)
                assert expected==computed_value

    def test_literal_eval_known_values(self):
        cases = [('from foo.bar import b; v = b + 1', {'foo.bar.b':0}, 1),
                 ('v = b + 1', {'b':0}, 1),
                 ('import sys; v = sys.version_info[:2]', {'sys.version_info':(3,7,18)}, (3,7)), 
                 ('from unknown import x; X=x; v = X + 1', {'m.X':0}, 1),
                 ('from unknown import x; X=x; v = X + 1', {'unknown.x':0}, 1),
                 ]
        for code, known_values, expected in cases:
            with self.subTest(code):
                proj = Project()
                proj.add_module(ast.parse(code), 'm')
                proj.analyze_project()
                v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
                computed_value = proj.state.literal_eval(v.node, known_values=known_values)
                assert expected==computed_value

    def test_literal_eval_fails(self):
        cases = [('class C:...\nv = C'),
                 ('def f():...\nv = f'),
                 ('v = lambda x:None'),
                 ('class C:...\nv = C()'),
                 ('v = {}'), # does not support dicts at the moment
                 ('v = {[1,2,3],}'), # a set of lists is invalid python code
                 ('v = {[1,2,3]:True}'),
                 ('v.attr = 1'), # attributes not supported - 87-88
                 ('for _ in []:\n m=m\nv = m'), # cyclic definition - 106
                 ('v = (1,(1,(1,(1,(1,(1))))))'), # expression is too complex - 108
                 ('x = 2;v = x.real'), # attribute access is only supported on modules and classdef - 122
                 ('x = 2;v=1;v+=x'), # augmented assignments are not supported - 137
                 ('v:int'), # cannot evaluate, it has no value - 142
                 ('(a,b)=[1,2]'), # cannot evaluate, it has no value - 318
                 ('a=1.5;v=[*(1,2,3), *a]'), # starred value is not iterable - 326
                 ('v = []+1'), # error in operation - 355
                 ('v = []>1'), # error in comparator - 376
                 ('v[1] = 1'), # subscripts store not supported - 389
                 ('v = 1[1]'), # not subscriptable - 389
                 ('v = -[1]'), # error in unary operator
                 ]
        try:
            _LiteralEval._MAX_JUMPS = 4
            for code in cases:
                with self.subTest(code):
                    proj = Project()
                    m = proj.add_module(ast.parse(code), 'm')
                    proj.analyze_project()
                    a = m.node.body[-1]
                    if isinstance(a, ast.Assign):
                        vnode = a.targets[0]
                    elif isinstance(a, (ast.AnnAssign, ast.AugAssign)):
                        vnode = a.target
                    else:
                        assert False
                    with self.assertRaises(StaticException):
                        proj.state.literal_eval(vnode, raise_on_ambiguity=True)
        finally:
            _LiteralEval._MAX_JUMPS = _ASTEval._MAX_JUMPS

    
    def test_literal_eval_imported_name(self):
        proj = Project()
        proj.add_module(ast.parse('from n.a.b.c import x;v = x'), 'm')
        proj.add_module(ast.parse('x = 1'), 'n.a.b.c')
        proj.analyze_project()
        v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
        with self.assertRaises(StaticUnknownValue, msg='unknown value: n.a.b.c.x'):
            proj.state.literal_eval(v.node)
        assert proj.state.literal_eval(v.node, follow_imports=True) == 1
