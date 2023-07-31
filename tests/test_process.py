import ast
from typing import Optional
from unittest import TestCase

from libstatic.model import Project, State
from libstatic.process import Processor, _ProcessingState
from libstatic._lib.imports import ParseImportedNames
from libstatic._lib.ancestors import Ancestors

class Cycle(Exception):...

class ImportProcessor(Processor[ast.Module, ast.Module]):

    def __init__(self, state:State) -> None:
        super().__init__()
        self._state = state
        self._getProcessedModuleCalls = []
    
    def getProcessedModule(self, name:str) -> Optional[ast.Module]:
        self._getProcessedModuleCalls.append(name)
        return super().getProcessedModule(name)
    
    def getModule(self, name: str) -> 'ast.Module | None':
        mod = self._state.get_module(name)
        if mod:
            return mod.node
        return None

    def processModule(self, astmod:ast.Module) -> ast.Module:
        modname = self._state.get_def(astmod).name()
        imports = ParseImportedNames(modname, is_package=False).visit_Module(astmod)
        ancestors = Ancestors()
        ancestors.visit(astmod)
        for al, imp in imports.items():
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ancestors.parents[al]):
                pass
            else:
                if self.state[self.getProcessedModule(imp.orgmodule)] is _ProcessingState.PROCESSING:
                    raise Cycle(f'{modname} <-> {imp.orgmodule}')
        return astmod

class TestProcessor(TestCase):
    
    def test_single_module(self):
        code = "import sys;x = 1"
        node = ast.parse(code)
        project = Project()
        project.add_module(node, 'mod1')
        
        ob = ImportProcessor(project.state)
        ob.process(m.node for m in project.state.get_all_modules())
        
        assert ob._getProcessedModuleCalls == ['sys']
        self.assertEqual(ob.processing_modules, [])
        self.assertEqual(list(ob.unprocessed_modules), [])
        self.assertEqual(len(ob.result), 1)
        self.assertIs(ob.result[project.state.get_module('mod1').node], node)
    
    def test_cyclic_import(self):
        code1 = "import mod2"
        code2 = "import mod1"
        project = Project()
        mod1=project.add_module(ast.parse(code1), 'mod1')
        mod2=project.add_module(ast.parse(code2), 'mod2')
        
        ob = ImportProcessor(project.state)
        with self.assertRaises(Cycle):
            ob.process(m.node for m in project.state.get_all_modules())
        
        assert ob._getProcessedModuleCalls == ['mod2', 'mod1']
        self.assertEqual(ob.processing_modules, [mod1.node, mod2.node])
        self.assertEqual(list(ob.unprocessed_modules), [mod1.node, mod2.node])
        self.assertEqual(len(ob.result), 0)
        
    def test_processModuleAST(self):
        code1 = "x = 1"
        node = ast.parse(code1)
        project = Project()
        mod1 = project.add_module(node, 'mod1')
        
        ob = ImportProcessor(project.state)
        ob.process(m.node for m in project.state.get_all_modules())
        
        self.assertEqual(ob.processing_modules, [])
        self.assertEqual(list(ob.unprocessed_modules), [])
        self.assertEqual(len(ob.result), 1)
        self.assertEqual(ob.result[mod1.node], node)
