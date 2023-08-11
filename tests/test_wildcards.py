from contextlib import redirect_stdout
from io import StringIO
import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project

class TestWildcardParsing(TestCase):
    def test_simple(self):
        mod1 = '''
        __all__ = []
        __all__ = __all__ + ['a', 'b'] 
        '''
        mod2 = '''
        from mod1 import *
        from mod1 import __all__ as _mod1all
        __all__ = _mod1all + ['c']
        '''
        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        proj.analyze_project()

        assert proj.state.get_dunder_all(m1)==['a', 'b']
        assert proj.state.get_dunder_all(m2)==['a', 'b', 'c']

    def test_defined_from_wildcard_import(self):
        mod1 = '''
        __all__ = ('b', 'c')
        '''
        mod2 = '''
        from mod1 import *
        from mod1 import __all__
        '''
        mod3 = '''
        from mod2 import __all__ as _a
        __all__ = ('z',)+_a
        '''
        # to get this right, we need to:
        # - resolve trivial modules (modules with no wildcard imports) first.
        # - patch the locals so that wildcard are replaced 

        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        proj.analyze_project()

        assert proj.state.get_all_names(m3)==('z', 'b', 'c')
        assert proj.state.get_all_names(m2)==('b', 'c')
        assert proj.state.get_all_names(m1)==('b', 'c')
    
    def test_defined_from_wildcard_import_invalid(self):
        mod1 = '''
        class c:...
        def stuff():...
        __all__ = ('b', c, stuff())
        '''
        mod2 = '''
        from mod1 import *
        from mod1 import __all__
        '''
        mod3 = '''
        from mod2 import __all__ as _a
        __all__ = ('z',)+_a
        '''
        out = StringIO()
        proj = Project(outstream=out)
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        proj.analyze_project()

        assert out.getvalue().replace('_ast', 'ast') == ("mod1:4:16: Cannot evaluate tuple element at index 1: ast.ClassDef at ?:2: Expected literal, got: ClassDef\n"
                                                         "mod1:4:19: Cannot evaluate tuple element at index 2: ast.Call at mod1:4:19: Unsupported node type\n")


        assert proj.state.get_all_names(m3)==('z', 'b',)
        assert proj.state.get_all_names(m2)==('b', )
        assert proj.state.get_all_names(m1)==('b', )
    
    def test_one_iteration(self):
        mod1 = '''
        __all__ = ['a', 'b']
        '''
        mod2 = '''
        from mod1 import __all__
        __all__ = __all__ + ['c']
        '''
        mod3 = '''
        from mod2 import __all__ as _mod2all
        __all__ = _mod2all + ['d']
        '''
        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        
        stdout = StringIO()
        with redirect_stdout(stdout):
            proj.analyze_project()
        assert proj.state.get_dunder_all(m3) == ['a', 'b', 'c', 'd']
        assert proj.state.get_dunder_all(m2) == ['a', 'b', 'c',]
        assert proj.state.get_dunder_all(m1) == ['a', 'b',]
        assert not stdout.getvalue()
    
    def test_still_one_iteration(self):
        mod1 = '''
        def foo():
            pass

        class Bar:
            pass

        class _Baz:
            pass

        __all__ = ['foo', 'Bar'] # <-
        '''

        mod2 = '''
        from mod1 import *
        from mod3 import Qux

        def spam():
            pass

        class Eggs:
            pass

        __all__ = ['spam', 'Eggs', 'Qux'] # <-
        '''

        mod3 = '''
        from mod4 import *

        # this Qux overrides the Qux from the wildcard
        class Qux:
            pass
        
        # ['Qux']
        '''

        mod4 = '''
        class Qux:
            pass

        __all__ = ['Qux', '_Quux'] # <-
        '''

        mod5 = '''
        from mod6 import *

        _Quux = 42
        '''

        mod6 = '''
        from mod4 import *

        class _Quux:
            pass
        
        # public names are ['Qux']
        '''

        mod7 = '''
        from mod2 import * # ['spam', 'Eggs', 'Qux']
        from mod6 import *

        def ham():
            pass
            
        # public names are ['spam', 'Eggs', 'Qux', 'ham']
        '''

        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        m4 = proj.add_module(ast.parse(dedent(mod4)), 'mod4')
        m5 = proj.add_module(ast.parse(dedent(mod5)), 'mod5')
        m6 = proj.add_module(ast.parse(dedent(mod6)), 'mod6')
        m7 = proj.add_module(ast.parse(dedent(mod7)), 'mod7')
        # proj.analyze_project()

        stdout = StringIO()
        with redirect_stdout(stdout):
            proj.analyze_project()
        out = stdout.getvalue()
        print(out)
        
        assert proj.state.get_all_names(m7) == ['ham', 'Qux', 'spam', 'Eggs']
        assert proj.state.get_all_names(m6) == ['Qux',]
        assert proj.state.get_all_names(m5) == ['Qux',]
        assert proj.state.get_all_names(m4) == ['Qux', '_Quux',]
        assert proj.state.get_all_names(m3) == ['Qux',]
        assert proj.state.get_all_names(m2) == ['spam', 'Eggs', 'Qux']

        assert not out

    def test_still_one_iteration_again(self):
        mod1 = '''
        __all__ = ['a', 'b']
        '''
        mod2 = '''
        from mod1 import __all__
        from mod1 import *
        '''
        mod3 = '''
        __all__ = ['c']
        '''
        mod4 = '''
        from mod2 import __all__
        from mod2 import *
        import mod3
        __all__ = __all__ + ['d'] + mod3.__all__
        '''
        mod5 = '''
        from mod4 import __all__
        from mod4 import *
        __all__ = __all__ + ['e']
        '''
        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        m4 = proj.add_module(ast.parse(dedent(mod4)), 'mod4')
        m5 = proj.add_module(ast.parse(dedent(mod5)), 'mod5')
        stdout = StringIO()
        with redirect_stdout(stdout):
            proj.analyze_project()
        
        assert proj.state.get_dunder_all(m5) == ['a', 'b', 'd', 'c', 'e']
        assert not stdout.getvalue()

    def test_unsupported(self):
        mod1 = '''
        __all__ = ['a', 'b']
        '''
        mod2 = '''
        c = d = True
        __all__ = ['d']
        '''
        mod3 = '''
        from mod1 import *
        from mod2 import *
        supposed_to_be_private = True
        __all__ = [_ for _ in dir() if not _.startswith("_")]
        '''

        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')

        proj.analyze_project()
        assert proj.state.get_all_names(m3) == ['supposed_to_be_private', 'd', 'a', 'b']
