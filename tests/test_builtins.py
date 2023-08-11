import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project, Def, _load_typeshed_mod_spec
from libstatic.exceptions import StaticNameError
from libstatic._lib.chains import defuse_chains_and_locals

def location(node:ast.AST, filename:str) -> str:
    return StaticNameError(node, filename=filename).location()

class TestBuiltins(TestCase):
    def test_real_builtins_module(self, ):
        proj = Project()
        proj.add_typeshed_module('builtins')
        proj.analyze_project()

        assert len(list(proj.state.goto_references(
            proj.state.get_defs_from_qualname('builtins.str')[0])))>200
    
    def test_builtin_name_chains(self):
        # Test for 
        # https://github.com/serge-sans-paille/beniget/pull/73
        code = '''
        import sys
        class property:...
        if sys.version_info >= (3, 11):
            class ExceptionGroup(Exception):
                @property
                def exceptions(self) -> tuple: ...
        '''
        mod = ast.parse(dedent(code))
        chains, locals, bchains, = defuse_chains_and_locals(mod, 
                                    'builtins', 'buitlins', False)
        property_def = next(iter(locals[mod]['property']))
        assert isinstance(property_def, Def)
        assert property_def.islive
        assert bchains['property'].islive

    def test_builtin_name_chains_real_builtins(self):
        path,mod,_ = _load_typeshed_mod_spec('builtins', None)
        chains, locals, bchains, = defuse_chains_and_locals(mod, 
                                    'builtins', 'buitlins', False)
        property_def = next(iter(locals[mod]['property']))
        assert isinstance(property_def, Def)
        assert property_def.islive
        assert bchains['property'].islive

    def test_references_builtins(self, ):
        builtins = '''
        from typing import Callable, Any
        class type:
            @property
            def __base__(self) -> type: ...
        class property:
            def getter(self, __fget: Callable[[Any], Any]) -> property: ...
        '''

        src1 = '''
        @property
        def f():
            ...
        '''

        proj = Project()
        b = proj.add_module(ast.parse(dedent(builtins)), 'builtins')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.analyze_project()

        assert list(proj.state.goto_references(proj.state.get_local(src1, 'f')[0])) == []

        property_func = proj.state.get_local(b, 'property')[0]
        references = list(proj.state.goto_references(property_func))
        assert len(references) == 3, [location(d.node, proj.state.get_filename(d.node)) for d in references]
    
    def test_builtins_as_dep(self):

        builtins = '''
        class property:...
        '''

        src1 = '''
        import builtins
        @builtins.property
        def f():...
        @property
        def g():...
        '''

        proj = Project(builtins=False)
        proj.add_module(ast.parse(dedent(builtins)), 'builtins')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.analyze_project()

        deco = src1.node.body[-1].decorator_list[0]
        property_func = proj.state.get_defs_from_qualname('builtins.property')[0]
        assert property_func is proj.state.goto_definition(deco)

        mod = proj.state.get_module('builtins')
        assert [location(u.node, proj.state.get_filename(u.node)) for u in mod.users()]== ['ast.alias at src1:2:7']
        assert [r for r in proj.state.goto_references(mod) if proj.state.get_root(r) is src1]
        assert proj.state.goto_references(property_func)
        references = [r for r in proj.state.goto_references(property_func) if proj.state.get_root(r) is src1]
        assert len(references) == 2, [location(d.node, proj.state.get_filename(d.node)) for d in references]
    
    def test_builtins_as_dep_real_builtins(self):

        src1 = '''
        import builtins
        @builtins.property
        def f():...
        @property
        def g():...
        '''

        proj = Project(builtins=True)
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.analyze_project()

        deco = src1.node.body[-1].decorator_list[0]
        property_func = proj.state.get_defs_from_qualname('builtins.property')[0]
        assert property_func is proj.state.goto_definition(deco)

        mod = proj.state.get_module('builtins')
        builtins_use = next(iter(u.node for u in mod.users()))
        assert location(builtins_use, proj.state.get_filename(builtins_use)) == 'ast.alias at src1:2:7'
        builtins_name_use = next(iter(u.node for u in proj.state.get_def(builtins_use).users()))
        assert location(builtins_name_use, proj.state.get_filename(builtins_name_use)) == 'ast.Name at src1:3:1'
        
        assert [r.node for r in proj.state.goto_references(mod) if proj.state.get_root(r) is src1] == [builtins_name_use]
        # assert list(proj.state.goto_references(property_func))

        references = [r for r in proj.state.goto_references(property_func) if proj.state.get_root(r) is src1]
        assert len(references) == 2, [location(d.node, proj.state.get_filename(d.node)) for d in references]
    
    
    def test_reachable_defs(self, ):
        builtins = '''
        import sys
        from typing import Callable, Any, Iterable, Generic, Iterator, overload
        tuple = bool = 1
        if sys.version_info >= (3, 10):
            class zip(Iterator[...], Generic[...]):...
        else:
            class zip(Iterator[...]):...
        '''

        src1 = '''
        zip
        '''

        proj = Project(python_version=(3, 11))
        proj.add_module(ast.parse(dedent(builtins)), 'builtins')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.analyze_project()
        zip_Defs = proj.state.get_local(proj.state.get_module('builtins'), 'zip')
        assert len(zip_Defs) == 2
        first_zip_new_def = zip_Defs[0]
        assert location(first_zip_new_def.node, proj.state.get_filename(first_zip_new_def.node)) == 'ast.ClassDef at builtins:6:4'
        second_zip_new_def = zip_Defs[1]
        assert not proj.state.is_reachable(second_zip_new_def.node)
        assert location(second_zip_new_def.node, proj.state.get_filename(second_zip_new_def.node)) == 'ast.ClassDef at builtins:8:4'
        assert proj.state.is_reachable(first_zip_new_def.node)
        
        zip_def = proj.state.goto_definition(src1.node.body[-1].value)
        assert location(zip_def.node, proj.state.get_filename(zip_def.node)) == 'ast.ClassDef at builtins:6:4'
        assert zip_def.islive
        assert proj.state.is_reachable(zip_def.node)