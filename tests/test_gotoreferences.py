import ast
from unittest import TestCase
from textwrap import dedent

from libstatic import Project

from . import location

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
    
    def test_goto_references_unreachable(self, ):

        for v, filter_unreachable in ([(3,4), False], [(3,8), False], [(3,11), False],
                                    [(3,11), True],[(3,8), True], [(3,4), True],):

            p = Project(python_version=v)
            
            dep = p.add_module(ast.parse(dedent('''
                import sys
                if sys.version_info > (3,7):
                    def deprecated(f):...
                else:
                    def deprecated(f):...
                    dep = deprecated
                ''')), 'deprecated')
            src1 = p.add_module(ast.parse(dedent('''\
                import sys
                if sys.version_info > (3,8):
                    from deprecated import deprecated
                else:
                    deprecated = lambda f:f
                @deprecated
                def f():...
                ''')), 'src1')
            _ = p.add_module(ast.parse(dedent('''\
                import src1
                @src1.deprecated
                class C:...
                from deprecated import deprecated
                @deprecated
                def f():...
                ''')), 'src2')
            
            p.analyze_project()
            first_dep_func, second_dep_func = p.state.get_local(dep, 'deprecated')
            
            if filter_unreachable:
                if v == (3,11):
                    assert p.state.is_reachable(first_dep_func) is True
                    assert p.state.is_reachable(second_dep_func) is False
                    legacy_func_ref = list(p.state.goto_references(second_dep_func, 
                                                                   filter_unreachable=True))
                    assert len(legacy_func_ref) == 1,  [location(r, p.state.get_filename(r)) for r in legacy_func_ref]
                    new_func_ref = list(p.state.goto_references(first_dep_func,
                                                                filter_unreachable=True))
                    
                    # TODO: the src1.deprecated is not recognized
                    # ...
                    deprecated_imp = p.state.get_local(src1, 'deprecated')[0]
                    assert p.state.is_reachable(deprecated_imp) is True
                    assert deprecated_imp.__class__.__name__ == 'Imp'
                    assert deprecated_imp.islive is True
                    assert p.state.goto_definition(deprecated_imp.node) is first_dep_func
                    # Should be 3 here
                    assert len(new_func_ref) == 2, [location(r, p.state.get_filename(r)) for r in new_func_ref]
                
                elif v == (3,8): 
                    assert p.state.is_reachable(first_dep_func) is True
                    assert p.state.is_reachable(second_dep_func) is False
                    legacy_func_ref = list(p.state.goto_references(second_dep_func, 
                                                                   filter_unreachable=True))
                    assert len(legacy_func_ref) == 1,  [location(r, p.state.get_filename(r)) for r in legacy_func_ref]
                    new_func_ref = list(p.state.goto_references(first_dep_func,
                                                                filter_unreachable=True))
                    assert len(new_func_ref) == 1, [location(r, p.state.get_filename(r)) for r in new_func_ref]
                
                elif v == (3,4):
                    assert p.state.is_reachable(first_dep_func) is False
                    assert p.state.is_reachable(second_dep_func) is True
                    legacy_func_ref = list(p.state.goto_references(second_dep_func, 
                                                                   filter_unreachable=True))
                    
                    assert len(legacy_func_ref) == 2, [location(r, p.state.get_filename(r)) for r in legacy_func_ref]
                    new_func_ref = list(p.state.goto_references(first_dep_func,
                                                                filter_unreachable=True))
                    assert len(new_func_ref) == 0,  [location(r, p.state.get_filename(r)) for r in new_func_ref]
                
                else:
                    assert False
            else:
            
                legacy_func_ref = list(p.state.goto_references(second_dep_func))
                assert len(legacy_func_ref) == 4
                # The goto_references() function does not filter 
                # unreachable definitions points by default.
                new_func_ref = list(p.state.goto_references(first_dep_func))
                assert len(new_func_ref) == 3
                # assert new_func_ref == legacy_func_ref

    def test_goto_references_killed(self, ):
        ...