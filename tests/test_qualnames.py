import ast
from typing import Optional
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project

class TestQualNames(TestCase):
    # the qualified name of Def 
    
    # in the static model, every NameDef has a qualname
    def test_function(self):
        src = '''
        lambda y: True
        def f():
            v = lambda x: True
        class cls:
            class Foo:
                def __init__(self):
                    t = [a for a in 'yes']
        '''

        p = Project()
        m = p.add_module(ast.parse(dedent(src)), 'test')
        p.analyze_project()
        s = p.state

        y = next(n for n in ast.walk(m.node) if isinstance(n, ast.arg) and n.arg=='y')
        f = next(n for n in ast.walk(m.node) if isinstance(n, ast.FunctionDef) and n.name=='f')
        v = next(n for n in ast.walk(m.node) if isinstance(n, ast.Name) and n.id=='v')
        v_val = next(n.value for n in ast.walk(m.node) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id=='v')
        x = next(n for n in ast.walk(m.node) if isinstance(n, ast.arg) and n.arg=='x')
        __init__ = next(n for n in ast.walk(m.node) if isinstance(n, ast.FunctionDef) and n.name=='__init__')
        t = next(n for n in ast.walk(m.node) if isinstance(n, ast.Name) and n.id=='t')
        t_val = next(n.value for n in ast.walk(m.node) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id=='t')
        a = next(n for n in ast.walk(m.node) if isinstance(n, ast.Name) and n.id=='a' and n.ctx.__class__.__name__ == 'Store')
        a_load = next(n for n in ast.walk(m.node) if isinstance(n, ast.Name) and n.id=='a' and n.ctx.__class__.__name__ == 'Load')

        assert s.get_qualname(s.get_def(y)) == 'test.<lambda>.<locals>.y'
        assert s.get_qualname(s.get_def(f)) == 'test.f'
        assert s.get_qualname(s.get_def(v)) == 'test.f.<locals>.v'
        assert s.get_qualname(s.get_def(v_val)) == 'test.f.<locals>.<lambda>'
        assert s.get_qualname(s.get_def(x)) == 'test.f.<locals>.<lambda>.<locals>.x'
        assert s.get_qualname(s.get_def(__init__)) == 'test.cls.Foo.__init__'
        assert s.get_qualname(s.get_def(t)) == 'test.cls.Foo.__init__.<locals>.t'
        assert s.get_qualname(s.get_def(t_val)) == 'test.cls.Foo.__init__.<locals>.<listcomp>'
        assert s.get_qualname(s.get_def(a)) == 'test.cls.Foo.__init__.<locals>.<listcomp>.<locals>.a'

class TestExpandExpr(TestCase):

    src = '''
        from twisted.ssl import ssl
        
        class session:
            from twisted.conch.interfaces import ISession
            I = ISession

        # not needed for test_expand_name
        session.I
        ssl.Stuff.object           
        session.ISession
        session.nosuchname
        nosuchname
        session
        123
        ISession()   
        '''
    
    def test_expand_expr(self) -> None:
        """
        The expand_expr() function finds a qualified name for
        a name expression in the AST.
        """

        node = ast.parse(dedent(self.src))

        p = Project()
        m = p.add_module(node, 'test')
        p.analyze_project()

        exprs = [n.value for n in m.node.body if isinstance(n, ast.Expr)]

        def lookup(expr: ast.expr) -> Optional[str]:
            return p.state.expand_expr(expr)

        # None is returned for non-name nodes.
        assert lookup(exprs[-1]) is None
        assert lookup(exprs[-2]) is None
        # Local names are returned with their full name.
        assert lookup(exprs[-3]) == 'test.session'
        # An unboud name results in a None value
        assert lookup(exprs[-4]) == None
        # Unknown names are resolved as far as possible.
        assert lookup(exprs[-5]) == 'test.session.nosuchname'
        assert lookup(exprs[-6]) == 'test.session.ISession'
        # Only the first name in the expression is resolvd
        # Imports are resolved
        assert lookup(exprs[-7]) == 'twisted.ssl.ssl.Stuff.object'
        assert lookup(exprs[-8]) == 'test.session.I'
    
    def test_expand_name(self) -> None:
        """
        The expand_expr() function finds a qualified name for
        a name expression in the AST.
        """

        node = ast.parse(dedent('''
        from twisted.ssl import ssl
        
        class session:
            from twisted.conch.interfaces import ISession
            I = ISession'''))

        p = Project()
        m = p.add_module(node, 'test')
        p.analyze_project()
        
        def lookup(name:str) -> Optional[str]:
            return p.state.expand_name(m, name)

        # Local names are returned with their qualnames
        assert lookup('session') == 'test.session'
        # An unboud name results in a None value
        assert lookup('unknown') == None
        # Unknown names are resolved as far as possible.
        assert lookup('session.nosuchname') == 'test.session.nosuchname'
        assert lookup('session.ISession') == 'test.session.ISession'
        # Only the first name in the expression is resolvd
        # Imports are resolved
        assert lookup('ssl.Stuff.object') == 'twisted.ssl.ssl.Stuff.object'
        assert lookup('session.I') == 'test.session.I'
        