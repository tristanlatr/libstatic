import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project
from libstatic.exceptions import StaticNameError, NodeLocation

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
    
    def test_goto_def_no_ambiguity(self):
        code = '''
        import sys
        if sys.version_info > (3,11):
            def f():...
        else:
            def f():...
        f
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,12), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()
        assert str(NodeLocation.make(proj.state.goto_def(node.body[-1].value, 
                                   raise_on_ambiguity=True), 'mod1')) == 'ast.FunctionDef at mod1:4:4'
    
    def test_get_attribute_no_ambiguity(self):
        code = '''
        import sys
        class c:
            if sys.version_info > (3,11):
                def f():...
            else:
                def f():...
        c.f
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,12), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()
        assert str(NodeLocation.make(proj.state.get_attribute(node.body[1], 'f')[-1], 'mod1')) == 'ast.FunctionDef at mod1:5:8'
        assert str(NodeLocation.make(proj.state.goto_definition(node.body[-1].value, 
                                   raise_on_ambiguity=True), 'mod1')) == 'ast.FunctionDef at mod1:5:8'

    def test_get_attribute_overloads(self):
        code = '''
        import sys
        from typing import overload
        class c:
            if sys.version_info > (3,11):
                @overload
                def f():...
                @overload
                def f():...
            else:
                @overload
                def f():...
                @overload
                def f():...
            
            f
        c.f
        '''
        node = ast.parse(dedent(code))
        proj = Project(python_version=(3,12), )
        proj.add_module(node, 'mod1')
        proj.analyze_project()
        assert str(NodeLocation.make(proj.state.goto_def(node.body[-2].body[-1].value, 
                                   raise_on_ambiguity=True), 'mod1')) == 'ast.FunctionDef at mod1:9:8'
        attrib, = proj.state.get_attribute(node.body[-2], 'f')
        assert str(NodeLocation.make(attrib, 'mod1')) == 'ast.FunctionDef at mod1:9:8'
        assert str(NodeLocation.make(proj.state.goto_definition(node.body[-1].value, 
                                   raise_on_ambiguity=True), 'mod1')) == 'ast.FunctionDef at mod1:9:8'

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
