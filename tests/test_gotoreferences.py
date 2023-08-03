import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project
from libstatic.exceptions import StaticNameError

def location(node:ast.AST, filename:str) -> str:
    return StaticNameError(node, filename=filename).location()

class TestReferences(TestCase):
    def test_goto_references(self, ):
        deprecated = '''
        def deprecated(f):
            return f
        '''

        src1 = '''
        from deprecated import deprecated
        @deprecated
        def f():
            ...
        '''

        src2 = '''
        from src1 import deprecated as dep
        import deprecated
        @dep
        class C:...
        '''

        src3 = '''
        import deprecated as dep
        @dep.deprecated
        class D:
            ...
        '''

        src4 = '''
        import src3, src2
        from src2 import dep
        @src3.dep.deprecated
        class F:...
        @src2.dep
        class G:...
        @src2.deprecated.deprecated
        class I:...
        @dep
        class H:...
        '''
        
        proj = Project()
        dep = proj.add_module(ast.parse(dedent(deprecated)), 'deprecated')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.add_module(ast.parse(dedent(src2)), 'src2')
        proj.add_module(ast.parse(dedent(src3)), 'src3')
        proj.add_module(ast.parse(dedent(src4)), 'src4')
        proj.analyze_project()

        assert list(proj.state.goto_references(proj.state.get_local(src1, 'f')[0])) == []

        deprecated_func = proj.state.get_local(dep, 'deprecated')[0]
        references = list(proj.state.goto_references(deprecated_func))
        
        # imports doesn't count as references.
        assert len(references) == 7, [location(d.node, proj.state.get_filename(d.node)) for d in references]