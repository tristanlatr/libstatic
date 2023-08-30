from unittest import TestCase

from libstatic import Project

class TestTypeshedLoading(TestCase):
    
    def test_builtins_module(self):
        proj = Project()
        builtinsmodule = proj.add_typeshed_module('builtins')
        assert len(list(proj.state.get_all_modules()))==1
        proj.analyze_project()
        assert proj.state.get_local(builtinsmodule, 'len')[-1].node.__class__.__name__=='FunctionDef'

    def test_builtins_module_load_dependencies(self):
        proj = Project(dependencies=1)
        builtinsmodule = proj.state.get_module('builtins')
        # assert len(proj._modules)>40
        proj.analyze_project()
        modlist = [mod.name() for mod in proj.state.get_all_modules()]
        assert 'typing' in modlist
        assert '_collections_abc' in modlist
        assert 'collections.abc' in modlist
        assert proj.state.get_local(builtinsmodule, 'len')[-1].node.__class__.__name__=='FunctionDef'
    