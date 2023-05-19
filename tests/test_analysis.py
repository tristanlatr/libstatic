from contextlib import redirect_stdout
from io import StringIO
import gast as ast
from unittest import TestCase
from textwrap import dedent

from beniget.beniget import Ancestors, Def
from libstatic.astutils import ast_repr
from libstatic.analysis import (GetStoredValue, Project, LiteralEval, ImportParser, 
                                ImportedName, OrderedBuilder, GotoDefinition, 
                                DeferredModuleException, ParseImportedNames, StateModifiers)

class TestUseDefChains(TestCase):
    def test_simple(self):
        code = '''
        import sys
        sys
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,7), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()

        sys_def = proj.state.goto_def(node.body[-1].value)
        assert isinstance(sys_def.node, ast.alias)

    def test_attr(self):
        code = '''
        import sys
        sys.version_info.major
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,7), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()

        sys_def = proj.state.goto_def(node.body[-1].value.value.value)
        assert isinstance(sys_def.node, ast.alias)

class TestUnreachable(TestCase):
    def test_simple_version_info_variable(self):
        code = '''
        import sys
        if sys.version_info.major==3:
            python3=True
        else:
            python3=False #unreachable
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,7), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()
        
        assign = node.body[-1].orelse[0]
        definition = node.body[-1].orelse[0].targets[0]

        assert definition in proj.state._unreachable

        # assert not proj.state.is_reachable(definition)
        # assert not proj.state.is_reachable(assign)
        
        # for n in ast.walk(assign):
        #     assert not proj.state.is_reachable(n)

    def test_simple_version_info_imports(self):
        code = '''
        import sys
        if sys.version_info < (3, 9):
            import importlib_resources
        else:
            import importlib.resources as importlib_resources
        '''

        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,7,0), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()
        
        stmt1 = node.body[-1].body[0]
        definition1 = stmt1.names[0]

        stmt2 = node.body[-1].orelse[0]
        definition2 = stmt2.names[0]

        assert ast_repr(stmt2)=='Import at <unknown>:6:4'
        assert ast_repr(definition2)=='alias at <unknown>:6:11'

        assert not proj.state.is_reachable(definition2)
        assert not proj.state.is_reachable(stmt2)

        assert proj.state.is_reachable(definition1)
        assert proj.state.is_reachable(stmt1)
        
        proj = Project(python_version=(3,10,0), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()

        assert proj.state.is_reachable(definition2)
        assert proj.state.is_reachable(stmt2)

        assert not proj.state.is_reachable(definition1)
        assert not proj.state.is_reachable(stmt1)

    code ='''
    if sys.version_info[:2] >= (3, 8):
        def stuff(f):
            ...
    else:
        def stuff(f):
            ...
    '''

class TestGetStoredValue(TestCase):
    def test_simple_assignment(self):
        code = 'a = 2'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = next((n for n in ast.walk(node) if isinstance(n, ast.Name)))
        constant = next((n for n in ast.walk(node) if isinstance(n, ast.Constant)))
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))
        
        assert GetStoredValue(name, assignment) is constant
        assert GetStoredValue(assignment.targets[0], assignment) is constant
    
    def test_no_value_assignment(self):
        code = 'a:int'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = next((n for n in ast.walk(node) if isinstance(n, ast.Name)))
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.AnnAssign)))
        
        assert GetStoredValue(name, assignment) is None
        assert GetStoredValue(assignment.target, assignment) is None
    
    def test_tuple_assignment(self):
        code = 'a,b = 2,3'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        constant = [n for n in ast.walk(node) if isinstance(n, ast.Constant)][0]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        assert GetStoredValue(name, assignment) is constant

        name = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        constant = [n for n in ast.walk(node) if isinstance(n, ast.Constant)][1]
        assert GetStoredValue(name, assignment) is constant

        tuple_ltarget = [n for n in ast.walk(node) if isinstance(n, ast.Tuple)][0]
        tuple_rvalue = [n for n in ast.walk(node) if isinstance(n, ast.Tuple)][1]
        assert GetStoredValue(tuple_ltarget, assignment) is tuple_rvalue
    
    def test_unsupported_assignment_star_value(self):
        code = 'd,e=(*a,)'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        d = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        e = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(d, assignment)
        
        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(e, assignment)

    def test_unsupported_nested_assignment(self):
        code = 'd,e,(f,g)=(1,2,(3,4))'

        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        d = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        e = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        assert GetStoredValue(d, assignment).__class__.__name__ == 'Constant'
        assert GetStoredValue(e, assignment).__class__.__name__ == 'Constant'

        f = [n for n in ast.walk(node) if isinstance(n, ast.Name)][2]
        g = [n for n in ast.walk(node) if isinstance(n, ast.Name)][3]

        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(f, assignment)
        
        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(g, assignment)

    def test_unsupported_assignment_unpack(self):
        code = 'a,b=c'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        a = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(a, assignment)

        code = '*a,c,b=(1,2,3,4,5,6)'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        a = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        c = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(a, assignment)
        with self.assertRaises(ValueError, msg='unsupported assignment'):
            GetStoredValue(c, assignment)

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
                 ('v = False or True or False', True),
                 ('v = True and True and False', False),
                 ('v = 0 or 4', 4),
                 ]
        for code, expected in cases:
            proj = Project()
            proj.add_module(ast.parse(code), 'm')
            proj.analyze_project()
            v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
            computed_value = LiteralEval(proj.state, v.node)
            assert expected==computed_value

    def test_literal_eval_known_values(self):
        cases = [('from foo.bar import b; v = b + 1', {'foo.bar.b':0}, 1),
                 ('v = b + 1', {'b':0}, 1),
                 ('import sys; v = sys.version_info[:2]', {'sys.version_info':(3,7,18)}, (3,7)), ]
        for code, known_values, expected in cases:
            proj = Project()
            proj.add_module(ast.parse(code), 'm')
            proj.analyze_project()
            v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
            computed_value = LiteralEval(proj.state, v.node, known_values=known_values)
            assert expected==computed_value

    def test_literal_eval_fails(self):
        cases = [('class C:...\nv = C'),
                 ('def f():...\nv = f'),
                 ('v = lambda x:None'),
                 ('class C:...\nv = C()'),
                 ('v = {}'), # does not support dicts at the moment
                 ('v = {[1,2,3],}'), # a set of lists is invalid python code
                 ('v = {[1,2,3]:True}'),
                 ]
        for code in cases:
            proj = Project()
            proj.add_module(ast.parse(code), 'm')
            proj.analyze_project()
            v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
            with self.assertRaises(ValueError):
                LiteralEval(proj.state, v.node)
    
    def test_literal_eval_fails_imported_name(self):
        proj = Project()
        proj.add_module(ast.parse('from n.a.b.c import x;v = x'), 'm')
        proj.add_module(ast.parse('x = 1'), 'n.a.b.c')
        proj.analyze_project()
        v = proj.state.get_local(proj.state.get_module('m'), 'v')[-1]
        with self.assertRaises(ValueError, msg='unknown value: n.a.b.c.x'):
            LiteralEval(proj.state, v.node)

class TestImportParser(TestCase):

    def test_import_parser(self):
        code = '''
        import mod2
        import pack.subpack
        import pack.subpack as a
        from mod2 import _k as k, _l as l, _m as m
        from pack.subpack.stuff import C
        '''
        expected = [{'mod2':(('mod2',),None)},
                    {'pack':(('pack',),None)},
                    {'a':(('pack','subpack'),None)},
                    {'k':(('mod2',),'_k'), 'l':(('mod2',),'_l'), 'm':(('mod2',),'_m')},
                    {'C':(('pack','subpack','stuff'),'C')},]
        parser = ImportParser('mod1', is_package=False)
        node = ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (ast.Import, ast.ImportFrom))
            for imp in parser.visit(import_node):
                assert isinstance(imp, ImportedName)
                assert imp.name() in expected_names
                expected_orgmodule, expected_orgname = expected_names[imp.name()]
                assert imp.orgmodule == expected_orgmodule
                assert imp.orgname == expected_orgname
                ran=True
        assert ran
    
    def test_import_parser_relative(self):
        code = '''
        from ...mod2 import bar as b
        from .pack import foo
        '''
        expected = [{'b':(('top','mod2'),'bar')},
                    {'foo':(('top','subpack','other','pack',),'foo')},]
        parser = ImportParser('top.subpack.other', is_package=True)
        node = ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (ast.Import, ast.ImportFrom))
            for imp in parser.visit(import_node):
                assert isinstance(imp, ImportedName)
                assert imp.name() in expected_names
                expected_orgmodule, expected_orgname = expected_names[imp.name()]
                assert imp.orgmodule == expected_orgmodule
                assert imp.orgname == expected_orgname
                ran=True
        assert ran

class TestDunderAllParsing(TestCase):
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
        # - do a first pass and resolve trivial modules:
        #   meaning modules with no wildcard imports
        # - patch the locals so that wildcard are replaced 

        proj = Project()
        m1 = proj.add_module(ast.parse(dedent(mod1)), 'mod1')
        m2 = proj.add_module(ast.parse(dedent(mod2)), 'mod2')
        m3 = proj.add_module(ast.parse(dedent(mod3)), 'mod3')
        proj.analyze_project()

        assert proj.state.get_public_names(m3)==('z', 'b', 'c')
        assert proj.state.get_public_names(m2)==('b', 'c')
        assert proj.state.get_public_names(m1)==('b', 'c')
    
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
        v = stdout.getvalue()
        assert 'done in 1 iterations' in v
    
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
        proj.analyze_project()

        stdout = StringIO()
        with redirect_stdout(stdout):
            proj.analyze_project()
        
        assert proj.state.get_public_names(m7) == ['ham', 'Qux', 'spam', 'Eggs']
        assert proj.state.get_public_names(m6) == ['Qux',]
        assert proj.state.get_public_names(m5) == ['Qux',]
        assert proj.state.get_public_names(m4) == ['Qux', '_Quux',]
        assert proj.state.get_public_names(m3) == ['Qux',]
        assert proj.state.get_public_names(m2) == ['spam', 'Eggs', 'Qux']

        v = stdout.getvalue()
        assert 'done in 1 iterations' in v

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
        proj.analyze_project()

        stdout = StringIO()
        with redirect_stdout(stdout):
            proj.analyze_project()
        assert proj.state.get_dunder_all(m5) == ['a', 'b', 'd', 'c', 'e']
        v = stdout.getvalue()
        assert 'done in 1 iterations' in v

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
        assert proj.state.get_public_names(m3) == ['supposed_to_be_private', 'd', 'a', 'b']

class TestGotoDefinition(TestCase):
    def test_import(self,):
        proj = Project()

        pack = proj.add_module(ast.parse(dedent('''
        def f():...
        ''')), name='pack', is_package=True)

        subpack = proj.add_module(ast.parse(dedent('''
        from external import E
        class C: ...
        ''')), name='pack.subpack')

        mod1 = proj.add_module(ast.parse(dedent('''
        import mod2
        # Importing pack.subpack imports pack also, but not if we use an alias
        import pack.subpack
        import pack.subpack as a
        pack
        a
        a.C
        mod2.k
        mod2.l
        mod2.m
        pack.subpack.C
        pack.subpack
        pack.subpack.E
        pack.f
        
        ''')), name='mod1')

        mod2 = proj.add_module(ast.parse(dedent('''
        k = 'fr'
        l = 3.14
        m = range(10)
        ''')), name='mod2')

        proj.analyze_project()

        body = [n.value for n in mod1.node.body if isinstance(n, ast.Expr)]

        # pack.f
        assert isinstance(GotoDefinition(proj.state, body[-1]), Def)

        # pack.subpack.E (externaly imported name)
        with self.assertRaises(LookupError):
            GotoDefinition(proj.state, body[-2])
        
        # pack.subpack
        assert subpack == proj.state.get_attribute(
            pack, 'subpack')[0] ==\
            proj.state.get_attribute(GotoDefinition(proj.state, pack.node), 'subpack', ignore_locals=True)[0]
        
        assert GotoDefinition(proj.state, body[-3]) == GotoDefinition(proj.state, subpack.node)

        # pack.subpack.C
        assert GotoDefinition(proj.state, body[-4]) == proj.state.get_attribute(
            GotoDefinition(proj.state, subpack.node), 'C')[0]
        
        # # mod2.m
        assert isinstance(GotoDefinition(proj.state, body[-5]).node, ast.Name)
        
        assert proj.state.goto_defs(mod1.node.body[0].names[0]) == [mod2]
        assert proj.state.goto_defs(body[-6].value) == [proj.state.get_def(mod1.node.body[0].names[0])]
        # mod2
        assert GotoDefinition(proj.state, body[-6].value) == mod2
        
        # mod2.l
        assert GotoDefinition(proj.state, body[-6]).name()=='l'
        
        # mod2.k
        assert GotoDefinition(proj.state, body[-7])

        # a.C
        assert GotoDefinition(proj.state, body[-8])

        # a
        assert GotoDefinition(proj.state, body[-9]) == GotoDefinition(proj.state, subpack.node)
        
        # pack
        assert GotoDefinition(proj.state, body[-10]) == GotoDefinition(proj.state, pack.node)
            
    def test_import_from(self, ):

        proj = Project()

        pack = proj.add_module(ast.parse(dedent('''
        from .subpack import C, E
        def f():...
        ''')), name='pack', is_package=True)

        subpack = proj.add_module(ast.parse(dedent('''
        from external import E
        class C: ...
        ''')), name='pack.subpack')

        mod1 = proj.add_module(ast.parse(dedent('''
        from mod2 import _k as k, _l as l, _m as m
        from pack import C, E, f
        k
        l
        m
        C
        E
        f
        
        ''')), name='mod1')

        mod2 = proj.add_module(ast.parse(dedent('''
        _k = 'fr'
        _l = ('i', 'j')
        _m = range(10)
        ''')), name='mod2')

        proj.analyze_project()

        filtered_body = [stmt.value for stmt in mod1.node.body if isinstance(stmt, ast.Expr)]

        # f
        assert GotoDefinition(proj.state, filtered_body[-1]).node is proj.state.get_local(pack, 'f')[-1].node
        
        # E
        # assert list(infer(filtered_body[-2])) == [Uninferable]
        with self.assertRaises(LookupError):
            GotoDefinition(proj.state, filtered_body[-2])
        
        # C
        assert GotoDefinition(proj.state, filtered_body[-3]).node  is proj.state.get_local(subpack, 'C')[-1].node
        
        # m
        m = GotoDefinition(proj.state, filtered_body[-4]).node.__class__.__name__ = 'Name'
        
        # l
        import_alias = proj.state.get_local(mod1,'l')[-1]
        assert isinstance(import_alias.node, ast.alias)
        
        GotoDefinition(proj.state, import_alias.node).node.__class__.__name__=='Name'

        # k
        assert GotoDefinition(proj.state, filtered_body[-6]).node.__class__.__name__ == 'Name'

def process_imports(proj:Project, build:OrderedBuilder, astmod:ast.Module):
    imports = ParseImportedNames(proj.state.get_def(astmod).name(), is_package=False).visit_Module(astmod)
    ancestors = Ancestors()
    ancestors.visit(astmod)
    for imp in imports.values():
        try:

            # if this import is inside a class or function, ignore it.
            # could be tweaked depending on the builder iteration.
            ancestors.parentInstance(imp.node, (ast.FunctionDef, 
                                                ast.AsyncFunctionDef, 
                                                ast.ClassDef))
        except ValueError:
            if not build.is_processed(build.get_processed_module(imp.orgmodule)):
                raise DeferredModuleException(imp.orgmodule)

class TestOrderedBuilder(TestCase):
    
    def test_single_module(self):
        code = "import sys;x = 1"
        node = ast.parse(code)
        project = Project()
        project.add_module(node, 'mod1')
        
        ob = OrderedBuilder(project.state, lambda n: n)
        ob.build()
        
        self.assertEqual(ob._processing_modules, [])
        self.assertEqual(list(ob._unprocessed_modules), [])
        self.assertEqual(len(ob._result), 1)
        self.assertIs(ob._result[project.state.get_module('mod1')], node)
    
    def test_cyclic_import(self):
        code1 = "import mod2"
        code2 = "import mod1"
        project = Project()
        mod1=project.add_module(ast.parse(code1), 'mod1')
        mod2=project.add_module(ast.parse(code2), 'mod2')
        
        ob = OrderedBuilder(project.state, lambda n: process_imports(project, ob, n) or n, max_iterations=100)
        with self.assertRaises(ValueError):
            ob.build()
        
        self.assertEqual(ob._processing_modules, [])
        self.assertEqual(list(ob._unprocessed_modules), [mod1, mod2])
        self.assertEqual(len(ob._result), 0)
        
    def test_processModuleAST(self):
        code1 = "x = 1"
        node = ast.parse(code1)
        project = Project()
        mod1 = project.add_module(node, 'mod1')
        
        def process(node):
            return node.body
        
        ob = OrderedBuilder(project.state, process)
        ob.build()
        
        self.assertEqual(ob._processing_modules, [])
        self.assertEqual(list(ob._unprocessed_modules), [])
        self.assertEqual(len(ob._result), 1)
        self.assertEqual(ob._result[mod1], node.body)

class TestTypeshedLoading(TestCase):
    
    def test_builtins_module(self):
        proj = Project()
        builtinsmodule = StateModifiers(proj.state).add_typeshed_module('builtins')
        assert len(list(proj.state.get_all_modules()))==1
        proj.analyze_project()
        assert proj.state.get_local(builtinsmodule, 'len')[-1].node.__class__.__name__=='FunctionDef'

    def test_builtins_module_load_dependencies(self):
        proj = Project(nested_dependencies=1)
        builtinsmodule = StateModifiers(proj.state).add_typeshed_module('builtins')
        # assert len(proj._modules)>40
        proj.analyze_project()
        modlist = [mod.name() for mod in proj.state.get_all_modules()]
        assert 'typing' in modlist
        assert '_collections_abc' in modlist
        assert 'collections.abc' in modlist
        assert proj.state.get_local(builtinsmodule, 'len')[-1].node.__class__.__name__=='FunctionDef'
    
class TestDumpProject(TestCase):
    def test_dump_load(self):
        code = "import sys;x = 1"
        node = ast.parse(code)
        project = Project()
        project.add_module(node, 'mod1')
        
        # no need to analyze project to dump it: it dumps the AST
        data = project.state.dump()
        assert isinstance(data, list)
        
        # there is one module
        assert len(data)==1
        mod = data[0]
        assert mod['is_package']==False
        assert mod['modname']=='mod1'
        assert mod['node']=={'_type': 'Module', 'body': [{'_type': 'Import', 'col_offset': 0, 'end_col_offset': 10, 'end_lineno': 1, 'lineno': 1, 'names': [{'_type': 'alias', 'asname': None, 'col_offset': 7, 'end_col_offset': 10, 'end_lineno': 1, 'lineno': 1, 'name': 'sys'}]}, {'_type': 'Assign', 'col_offset': 11, 'end_col_offset': 16, 'end_lineno': 1, 'lineno': 1, 'targets': [{'_type': 'Name', 'col_offset': 11, 'ctx': {'_type': 'Store'}, 'end_col_offset': 12, 'end_lineno': 1, 'id': 'x', 'lineno': 1}], 'type_comment': None, 'value': {'_type': 'Constant', 'col_offset': 15, 'end_col_offset': 16, 'end_lineno': 1, 'kind': None, 'lineno': 1, 'n': 1, 's': 1, 'value': 1}}], 'type_ignores': []}

        new_proj = Project()
        StateModifiers(new_proj.state).load(data)

        assert project.state.dump() == new_proj.state.dump()
        new_proj.analyze_project()
        assert LiteralEval(new_proj.state, new_proj.state.get_local(new_proj.state.get_module('mod1'), 'x')[-1].node)==1