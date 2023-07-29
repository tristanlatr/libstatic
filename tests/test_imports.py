import ast
from unittest import TestCase
from textwrap import dedent

from libstatic.shared import ast_node_name
from libstatic.imp import ImportParser

class TestImportParser(TestCase):

    def test_import_parser(self):
        code = '''
        import mod2
        import pack.subpack
        import pack.subpack as a
        from mod2 import _k as k, _l as l, _m as m
        from pack.subpack.stuff import C
        '''
        expected = [{'mod2':('mod2', None)},
                    {'pack':('pack' ,None)},
                    {'a':('pack.subpack', None)},
                    {'k':('mod2','_k'), 
                     'l':('mod2','_l'), 
                     'm':('mod2','_m')},
                    {'C':('pack.subpack.stuff','C')},]
        parser = ImportParser('mod1', is_package=False)
        node = ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (ast.Import, ast.ImportFrom))
            for al,(orgmodule, orgname) in parser.visit(import_node).items():
                assert ast_node_name(al) in expected_names
                expected_orgmodule, expected_orgname = expected_names[ast_node_name(al)]
                assert orgmodule == expected_orgmodule
                assert orgname == expected_orgname
                ran=True
        assert ran
    
    def test_import_parser_relative(self):
        code = '''
        from ...mod2 import bar as b
        from .pack import foo
        '''
        expected = [{'b':('top.mod2','bar')},
                    {'foo':('top.subpack.other.pack','foo')},]
        parser = ImportParser('top.subpack.other', is_package=True)
        node = ast.parse(dedent(code))
        assert len(expected)==len(node.body)
        for import_node, expected_names in zip(node.body, expected):
            assert isinstance(import_node, (ast.Import, ast.ImportFrom))
            for al,(orgmodule, orgname) in parser.visit(import_node).items():
                assert ast_node_name(al) in expected_names
                expected_orgmodule, expected_orgname = expected_names[ast_node_name(al)]
                assert orgmodule == expected_orgmodule
                assert orgname == expected_orgname
                ran=True
        assert ran


