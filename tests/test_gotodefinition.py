import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Def, Project
from libstatic.exceptions import StaticImportError, StaticAmbiguity


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
        assert isinstance(proj.state.goto_definition(body[-1]), Def)

        # pack.subpack.E (externaly imported name)
        with self.assertRaises(StaticImportError):
            proj.state.goto_definition(body[-2])
        
        # pack.subpack
        assert subpack == proj.state.get_attribute(
            pack, 'subpack')[0] ==\
            proj.state.get_attribute(proj.state.goto_definition(pack.node), 'subpack', ignore_locals=True)[0]
        
        assert proj.state.goto_definition(body[-3]) == proj.state.goto_definition(subpack.node)

        # pack.subpack.C
        assert proj.state.goto_definition(body[-4]) == proj.state.get_attribute(
            proj.state.goto_definition(subpack.node), 'C')[0]
        
        # # mod2.m
        assert isinstance(proj.state.goto_definition(body[-5]).node, ast.Name)
        
        assert proj.state.goto_defs(mod1.node.body[0].names[0]) == [mod2]
        assert proj.state.goto_defs(body[-6].value) == [proj.state.get_def(mod1.node.body[0].names[0])]
        # mod2
        assert proj.state.goto_definition(body[-6].value) == mod2
        
        # mod2.l
        assert proj.state.goto_definition(body[-6]).name()=='l'
        
        # mod2.k
        assert proj.state.goto_definition(body[-7])

        # a.C
        assert proj.state.goto_definition(body[-8])

        # a
        assert proj.state.goto_definition(body[-9]) == proj.state.goto_definition(subpack.node)
        
        # pack
        assert proj.state.goto_definition(body[-10]) == proj.state.goto_definition(pack.node)
            
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
        assert proj.state.goto_definition(filtered_body[-1]).node is proj.state.get_local(pack, 'f')[-1].node
        
        # E
        with self.assertRaises(StaticImportError, msg='Import target not found: E'):
            proj.state.goto_definition(filtered_body[-2])
        
        # C
        assert proj.state.goto_definition(filtered_body[-3]).node  is proj.state.get_local(subpack, 'C')[-1].node
        
        # m
        m = proj.state.goto_definition(filtered_body[-4]).node.__class__.__name__ = 'Name'
        
        # l
        import_alias = proj.state.get_local(mod1,'l')[-1]
        assert isinstance(import_alias.node, ast.alias)
        
        proj.state.goto_definition(import_alias.node).node.__class__.__name__=='Name'

        # k
        assert proj.state.goto_definition(filtered_body[-6]).node.__class__.__name__ == 'Name'

    def test_import_ambiguous(self, ):
        typing = '''
        class Protocol:...
        '''
        src1 = '''
        if something():
            from typing import Protocol
        if not something():
            class Protocol:...
        '''
        src2 = '''
        from src1 import Protocol
        Protocol
        '''
        proj = Project()
        proj.add_module(ast.parse(dedent(typing)), 'typing')
        proj.add_module(ast.parse(dedent(src1)), 'src1')
        mod2 = proj.add_module(ast.parse(dedent(src2)), 'test')
        proj.analyze_project()

        expr = mod2.node.body[-1].value
        with self.assertRaises(StaticAmbiguity):
            assert proj.state.goto_definition(expr, raise_on_ambiguity=True)
        assert proj.state.get_filename(proj.state.goto_definition(expr).node)=='typing'
    
    def test_aliases(self, ):
        src = '''
        class C:...
        constructor = C
        constructor
        '''

        proj = Project()
        mod = proj.add_module(ast.parse(dedent(src)), 'test')
        proj.analyze_project()

        expr = mod.node.body[-1].value
        assert proj.state.goto_definition(expr, 
                              raise_on_ambiguity=True,
                              follow_aliases=True).node.__class__.__name__ == 'ClassDef'
        assert proj.state.goto_definition(expr, 
                              raise_on_ambiguity=True).node.__class__.__name__ == 'Name'
    
    def test_aliases_ambiguous(self, ):
        src = '''
        class C:...
        if something():
            constructor = C
        if not something():
            class constructor:...
        constructor
        '''

        proj = Project()
        mod = proj.add_module(ast.parse(dedent(src)), 'test')
        proj.analyze_project()

        expr = mod.node.body[-1].value
        with self.assertRaises(StaticAmbiguity):
            assert proj.state.goto_definition(expr, 
                                raise_on_ambiguity=True)

    def test_no_imports_but_aliases(self, ):
        src = '''
        from x import b as B
        constructor = B
        constructor
        '''

        proj = Project()
        mod = proj.add_module(ast.parse(dedent(src)), 'test')
        proj.analyze_project()

        expr = mod.node.body[-1].value
        assert proj.state.goto_definition(expr, 
                              raise_on_ambiguity=True,
                              follow_aliases=True, 
                              follow_imports=False).node.__class__.__name__ == 'alias'
    
    def test_arguments(self, ):
        ...