import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.model import Project
from libstatic.exceptions import StaticNameError

def location(node:ast.AST, filename:str) -> str:
    return StaticNameError(node, filename=filename).location()

class TestBuiltins(TestCase):
    def test_real_builtins_module(self, ):
        proj = Project()
        proj.add_typeshed_module('builtins')
        proj.analyze_project()

        assert len(list(proj.state.goto_references(
            proj.state.get_defs_from_qualname('builtins.str')[0])))>200

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
    
    def test_reachable_defs(self, ):
        builtins = '''
        import sys
        from typing import Callable, Any, Iterable, Generic, Iterator, overload
        tuple = bool = 1
        
        class zip(Iterator[...], Generic[...]):
            if sys.version_info >= (3, 10):
                @overload
                def __new__(cls, __iter1: Iterable[...], *, strict: bool = ...) -> zip[tuple[...]]: ...
                @overload
                def __new__(cls, __iter1: Iterable[...], __iter2: Iterable[...], *, strict: bool = ...) -> zip[tuple[..., ...]]: ...
                @overload
                def __new__(
                    cls,
                    __iter1: Iterable[Any],
                    __iter2: Iterable[Any],
                    *iterables: Iterable[Any],
                    strict: bool = ...,
                ) -> zip[tuple[Any, ...]]: ...
            else:
                @overload
                def __new__(cls, __iter1: Iterable[...]) -> zip[tuple[...]]: ...
                @overload
                def __new__(
                    cls,
                    __iter1: Iterable[Any],
                    *iterables: Iterable[Any],
                ) -> zip[tuple[Any, ...]]: ...

            def __iter__(self) -> ...: ...
            def __next__(self) -> ...: ...
        '''

        src1 = '''
        zip.__new__
        '''

        proj = Project(python_version=(3, 11))
        proj.add_module(ast.parse(dedent(builtins)), 'builtins')
        src1 = proj.add_module(ast.parse(dedent(src1)), 'src1')
        proj.analyze_project()

        first_zip_def = proj.state.get_defs_from_qualname('builtins.zip.__new__')[0]
        # TODO: islive flag seems still True...
        # assert first_zip_def.islive == False, location(first_zip_def.node, proj.state.get_filename(first_zip_def.node))

        zip_def = proj.state.goto_definition(src1.node.body[-1].value, raise_on_ambiguity=True)
        # TODO: Should not be this one :/
        assert location(zip_def.node, proj.state.get_filename(zip_def.node)) == 'builtins:24:8'
        assert zip_def.islive
        assert proj.state.is_reachable(zip_def)