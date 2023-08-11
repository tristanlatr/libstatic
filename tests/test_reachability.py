import ast
from unittest import TestCase
from textwrap import dedent

from libstatic import Project

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

        assert not proj.state.is_reachable(definition)
        assert not proj.state.is_reachable(assign)
        
        for n in ast.walk(assign):
            assert not proj.state.is_reachable(n)

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
