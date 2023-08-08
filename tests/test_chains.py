import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project
from libstatic.exceptions import StaticNameError

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
    
    def test_arg(self):
        src = '''
        # uncomment me when https://github.com/serge-sans-paille/beniget/pull/70 is fixed
        # lambda x,args,kwargs: True
        def f(x:bool, *args, **kwargs):...
        '''

        node = ast.parse(dedent(src))
        proj = Project(python_version=(3,7), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()

        for a in (n for n in ast.walk(node) if isinstance(n, ast.arg)):
            fn = proj.state.get_enclosing_scope(a)
            assert isinstance(fn.node, (ast.FunctionDef, ast.Lambda))
            assert list(proj.state.get_locals(fn))==['x','args','kwargs']
            proj.state.get_def(a)


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
        
        # TODO: Change this to raising error
        version_def = proj.state.goto_def(node.body[-1].value.value)
        assert isinstance(version_def.node, ast.Name)
        assert version_def.node.id=='sys'
        major_def = proj.state.goto_def(node.body[-1].value)
        assert isinstance(major_def.node, ast.Attribute)
        assert major_def.node.attr=='version_info'

    def test_annassign(self):
        typing = '''
        Optional = object
        '''
        code = '''
        from typing import Optional
        foo: Unbound
        var: Optional = None
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,7), )
        proj.add_module(ast.parse(dedent(typing)), 'typing')
        proj.add_module(node, 'mod1')
        proj.analyze_project()

        optional_def = proj.state.goto_def(node.body[-1].annotation)
        assert isinstance(optional_def.node, ast.alias)
        assert all(isinstance(n.node, ast.alias) for n in proj.state.goto_defs(node.body[-1].annotation))

        with self.assertRaises(StaticNameError):
            proj.state.goto_def(node.body[-2].annotation)
