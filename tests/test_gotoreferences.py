import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project

class TestReferences(TestCase):
    def test_references_(self, ):
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
        @dep
        class C:
            def __init__(self):...
            v: dep[int] = None
        '''

        src3 = '''
        import deprecated as dep
        @dep.deprecated
        class D:
            ...
        '''
        
        proj = Project()
        dep = proj.add_module(ast.parse(dedent(deprecated)), 'deprecated')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.add_module(ast.parse(dedent(src2)), 'src2')
        proj.add_module(ast.parse(dedent(src3)), 'src3')
        proj.analyze_project()

        assert list(proj.state.goto_references(proj.state.get_local(src1, 'f')[0])) == []

        deprecated_func = proj.state.get_local(dep, 'deprecated')[0]
        references = list(proj.state.goto_references(deprecated_func))
        # imports doesn't count as references.
        assert len(references) == 3
        module_references = list(proj.state.goto_references(dep))
        assert len(module_references) == 1