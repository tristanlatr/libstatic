
import ast
from unittest import TestCase

from libstatic import Project

# class TestDummyDumpLoad(TestCase):

#     def test_dump_load(self):
#         code = "import sys;x = 1"
#         node = ast.parse(code)
#         project = Project()
#         project.add_module(node, 'mod1')
        
#         # no need to analyze project to dump it: it dumps the AST
#         data = project.state._dump()
#         assert isinstance(data, list)
        
#         # there is one module
#         assert len(data)==1
#         mod = data[0]
#         assert mod['is_package']==False
#         assert mod['modname']=='mod1'
#         assert mod['node']=={'_type': 'Module', 'body': [{'_type': 'Import', 'col_offset': 0, 'end_col_offset': 10, 'end_lineno': 1, 'lineno': 1, 'names': [{'_type': 'alias', 'asname': None, 'col_offset': 7, 'end_col_offset': 10, 'end_lineno': 1, 'lineno': 1, 'name': 'sys'}]}, {'_type': 'Assign', 'col_offset': 11, 'end_col_offset': 16, 'end_lineno': 1, 'lineno': 1, 'targets': [{'_type': 'Name', 'col_offset': 11, 'ctx': {'_type': 'Store'}, 'end_col_offset': 12, 'end_lineno': 1, 'id': 'x', 'lineno': 1}], 'type_comment': None, 'value': {'_type': 'Constant', 'col_offset': 15, 'end_col_offset': 16, 'end_lineno': 1, 'kind': None, 'lineno': 1, 'n': 1, 's': 1, 'value': 1}}], 'type_ignores': []}

#         new_proj = Project()
#         new_proj.state._load(data)

#         assert project.state._dump() == new_proj.state._dump()
#         new_proj.analyze_project()
#         assert new_proj.state.literal_eval( 
#                 new_proj.state.get_local(new_proj.state.get_module('mod1'), 'x')[-1].node)==1