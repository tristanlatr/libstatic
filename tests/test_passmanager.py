from __future__ import annotations

from collections import defaultdict
import sys
from textwrap import dedent
import time
from typing import Any, Collection, Mapping

from unittest import TestCase

import ast, inspect

# implementation details
from libstatic._lib.arguments import ArgSpec, iter_arguments # Yields all arguments of the given ast.arguments node as ArgSpec instances.
from libstatic._lib.assignment import get_stored_value # Given an ast.Name instance with Store context and it's parent assignment statement, figure out the right hand side expression that is stored in the symbol.
from libstatic._lib.exceptions import NodeLocation
from libstatic._lib.shared import LocalStmtVisitor

# main framework module we're testing
from libstatic._lib.passmanager import (PassManager, Module, NodeAnalysis, FunctionAnalysis, 
                                        ClassAnalysis, ModuleAnalysis, Transformation, Analysis, Pass)
from libstatic._lib import passmanager, analyses
from libstatic._lib.passmanager import events
from libstatic._lib.passmanager._astcompat import ASTCompat


# factory

def fromPasses(modules: Collection[Module], passes: Collection[type[Pass]]) -> PassManager:
    pm = PassManager()
    _transforms, _analyses = (), ()
    for p in passes:
        if p.isInterModules():
            raise TypeError('only intra-module passes can be run with this function')
        if issubclass(p, Transformation):
            _transforms += (p,)
        elif issubclass(p, Analysis):
            _analyses += (p,)

    for m in modules:
        pm.add_module(m)
        for t in _transforms:
            pm.apply(t, m.node)
        for a in _analyses:
            pm.gather(a, m.node)
    
    return pm


from time import perf_counter

class catchtime:
    def __init__(self, label=None):
        self.label = label or ''
        if label:
            self.label += ' '

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'{self.label}time: {self.time:.3f} seconds'
        print(self.readout)


# test analysis

class node_list(NodeAnalysis[list[ast.AST]], ast.NodeVisitor):
    "List of all children in the node recursively depth first"
    def visit(self, node: ast.AST) -> list[ast.AST]:
        self.result.append(node)
        super().visit(node)
        return self.result

    def doPass(self, node: ast.AST) -> list[ast.AST]:
        self.result = []
        self.visit(node)
        return self.result


class class_bases(ClassAnalysis[list[str]]):
    "The bases of a class as strings"
    def doPass(self, node: ast.ClassDef) -> list[str]:
        return [ast.unparse(n) for n in node.bases]

class function_arguments(FunctionAnalysis[list[ArgSpec]]):
    "List of function arguments"
    def doPass(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ArgSpec]:
        return list(iter_arguments(node.args))

class function_accepts_any_keywords(FunctionAnalysis[bool]):
    "Whether a function will accept any keyword argument, based on it's signature"
    
    dependencies = (function_arguments, )
    
    def doPass(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return any(a.kind == inspect.Parameter.VAR_KEYWORD for a in self.function_arguments)

class class_count(NodeAnalysis[int], LocalStmtVisitor):
    """
    Counts the number of classes in the locals of the given node.
    """
    
    def visit_ClassDef(self, node):
        self.result += 1
    
    def doPass(self, node: ast.Module) -> int:
        self.result = 0
        self.visit(node)
        return self.result

# simple test stuff

class simple_symbol_table(ModuleAnalysis[dict[str, list[ast.AST]]], ast.NodeVisitor):
    "A simbol table, for the module level only"
    def doPass(self, node: ast.Module) -> dict[str, list[ast.AST]]:
        self.result = defaultdict(list)
        self.generic_visit(node)
        return self.result
    
    def visit_Import(self, node:ast.Import):
        for al in node.names:
            name, asname = al.name.split('.')[0], al.asname
            self.result[asname or name].append(al)
    
    visit_ImportFrom = visit_Import

    def visit_ClassDef(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        self.result[node.name].append(node)

    visit_AsyncFunctionDef = visit_FunctionDef = visit_ClassDef

    def visit_Name(self, node: ast.Name):
        if not node.ctx.__class__.__name__ == 'Store':
            return
        self.result[node.id].append(node)

class simple_goto_def(NodeAnalysis['ast.AST | None']):
    "goto the definition of th symbol, but only works at module level"
    dependencies = (simple_symbol_table, analyses.ancestors)

    def doPass(self, node: ast.Name) -> ast.AST | None:
        if node.__class__.__name__ != 'Name':
            raise TypeError(node)
        if node.ctx.__class__.__name__ == 'Store':
            try:
                assign = next(n for n in self.ancestors[node] 
                                if n.__class__.__name__ in ('Assign', 'AnnAssign'))
                return get_stored_value(node, assign)
            except Exception as e:
                if __debug__:
                    print(str(e), file=sys.stderr)
                return None
        
        elif node.ctx.__class__.__name__ == 'Load':
            defi = self.simple_symbol_table[node.id]
            if defi: 
                return defi[-1]
            return None
        
        else:
            raise TypeError(node.ctx)


# test transforms

class transform_trues_into_ones(Transformation, ast.NodeTransformer):
    "True -> 1"
    preservesAnalyses = (class_count, )

    def visit_Constant(self, node):
        if node.value is True:
            self.update = True
            return ast.Constant(value=1)
        else:
            return node
    
    def doPass(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

class transform_trues_into_ones_opti(Transformation, ast.NodeTransformer):
    "True -> 1"
    preservesAnalyses = (class_count, )

    def visit_Constant(self, node):
        if node.value is True:
            self.recReplaceNode(node, newNode:=ast.Constant(value=1))
            return newNode
        else:
            return node
    
    def doPass(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

# An analysis that has a transform in it's dependencies

class has_optional_parameters(NodeAnalysis):
    optionalParameters = {'filter': False, 'inherited': False}
    def doPass(self, node): pass

class has_required_parameters(NodeAnalysis):
    requiredParameters = ('name', 'thing', )
    def doPass(self, node): pass

class has_both_required_and_optional_parameters(NodeAnalysis):
    requiredParameters = ('name', 'thing', )
    optionalParameters = {'filter': False, 'inherited': False}
    def doPass(self, node): pass

class requires_modules(NodeAnalysis):
    dependencies = (passmanager.modules, )
    def doPass(self, node): return 1

class has_dynamic_dependencies(NodeAnalysis):
    dependencies = (has_required_parameters, )
    optionalParameters = {'project_wide': False}
    @classmethod
    def prepareClass(cls):
        if cls.project_wide:
            cls.dependencies += (requires_modules, )
    def doPass(self, node): pass

class literal_ones_count(ModuleAnalysis[int], ast.NodeVisitor):
    "counts the number of literal '1'"
    dependencies = (transform_trues_into_ones, 
                    # this analysis is useless but it should not 
                    # crash because it's listed after the transform., even before is valid!
                    class_count) 
    
    def visit_Constant(self, node):
        if node.value is not True and node.value == 1:
            self.result += 1

    def doPass(self, node: ast.AST) -> int:
        self.result = 0
        self.visit(node)
        return self.result

# an invalid analysis because it has an analysis BEFORE a transformation

class still_valid_analysis_dependencies_order(ModuleAnalysis[int]):
    "this analysis is not valid actually, not everything is ok."
    dependencies = (node_list, transform_trues_into_ones)

    def doPass(self, node: ast.AST) -> int:
        return 1

# an invalid analysis because it has not doPass() method

class invalid_analysis_no_doPass_method(ModuleAnalysis[int]):
    ...


# test cases begins

class TestTestAnalysis(TestCase):
    
    def test_simple_symbol_table(self):
        src = ('class A(object): ...\n'
               'def f():...\n'
               'var = x = True\n'
               'from x import *\n'
               'import pydoctor.driver\n')
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        symtable = pm.gather(simple_symbol_table, pm.modules['test'].node)
        assert len(symtable) == 6
        assert list(symtable) == ['A', 'f', 'var', 'x', '*', 'pydoctor']
    
    def test_simple_goto_def(self):
        src = ('from x import y\n'
               'var = y\n'
               'var')
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        varuse = pm.modules['test'].node.body[-1].value
        defi0 = pm.gather(simple_goto_def, varuse)
        assert defi0.__class__.__name__ == 'Name'
        assert defi0.ctx.__class__.__name__ == 'Store'
        assert defi0.id == 'var'

        defi1 = pm.gather(simple_goto_def, defi0)
        assert defi1.__class__.__name__ == 'Name'
        assert defi1.ctx.__class__.__name__ == 'Load'
        assert defi1.id == 'y'

        defi2 = pm.gather(simple_goto_def, defi1)
        assert defi2.__class__.__name__ == 'alias'
        assert defi2.name == 'y'
    

class TestManagerKeepsTracksOfRootModules(TestCase):
    """
    This test case ensures that the PassManager.modules attribute is mutated
    when a transformation is applied.
    """

    def test_root_module(self):
        src = 'v = True'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        True_node = pm.modules['test'].node.body[0].value
        assert pm.modules[True_node].modname == 'test'

        updates, node = pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert updates
        
        assert pm.modules[node.body[0].value].modname == 'test'
        
        # since we use weakkeymapping the key will stay in the mapping until all references
        # are removed.
        assert True_node in pm.modules.ancestors
        assert True_node in pm.modules

        # But if we completely remove the module, then it will be marked as an error
        pm.remove_module(pm.modules['test'])
        with self.assertRaises(KeyError):
            pm.modules[True_node]


class TestPassManagerFramework(TestCase):
    
    def test_analysis_class_call_optional_parameters(self):
        # when calling a class with already the arguments default, this has not affect
        assert has_optional_parameters is has_optional_parameters(filter=False)
        assert has_optional_parameters is has_optional_parameters(filter=False, inherited=False)
        
        # this creates a new subclass
        assert has_optional_parameters is not has_optional_parameters(filter=True)
        assert has_optional_parameters(filter=True) is has_optional_parameters(filter=True)
                                                       # but this doesn't because it uses the default 
                                                       # for inherited and filter=True has already been created
        assert has_optional_parameters(filter=True) is has_optional_parameters(filter=True, inherited=False)
        
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        node = pm.modules['test'].node

        # these analysis can be run
        pm.gather(has_optional_parameters, node)
        pm.gather(has_optional_parameters(filter=True), node)
        pm.gather(has_optional_parameters(filter=True, inherited=True), node)

    def test_analysis_class_call_multiple_calls(self):
        # we can call the class several times it we like but that's creating again a new subclass and we can't do that...
        # TODO: fix it properly idk..

        with self.assertRaises(TypeError, msg='you must list all parameters into a single call to class has_optional_parameters'):
            has_optional_parameters(filter=True)(filter=False)
        
        with self.assertRaises(TypeError, msg='you must list all parameters into a single call to class has_both_required_and_optional_parameters'):
            has_both_required_and_optional_parameters(name='show')(filter=False)
        
        with self.assertRaises(TypeError, msg='analysis subclassing is not supported, create another analysis depending on this one'):
            class _subclassing_not_supported(has_both_required_and_optional_parameters):
                filter = True
            _subclassing_not_supported(name='do not subclass analyses!')

    def test_analysis_class_call_required_parameters(self):

        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        node = pm.modules['test'].node

        with self.assertRaises(TypeError):
            pm.gather(has_required_parameters, node)
        with self.assertRaises(TypeError):
            pm.gather(has_required_parameters(name='stuff'), node)
        with self.assertRaises(TypeError):
            pm.gather(has_required_parameters(thing='stuff'), node)
        with self.assertRaises(TypeError):
            pm.gather(has_both_required_and_optional_parameters(thing='stuff'), node)
        with self.assertRaises(TypeError, msg=''):
            pm.gather(has_both_required_and_optional_parameters(thing='stuff', inherited=True), node)

        pm.gather(has_required_parameters(thing='stuff', name='thing'), node)
        pm.gather(has_both_required_and_optional_parameters(thing='stuff', name='thing'), node)

    def test_modules_analysis(self):
        
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        pm.add_module(Module(ast.parse('import test'), 'test2'))
        node1 = pm.modules['test'].node
        node2 = pm.modules['test2'].node

        class list_modules_keys(NodeAnalysis):
            dependencies = (passmanager.modules, )
            def doPass(self, node: ast.AST) -> list[str]:
                return list(self.modules)
        
        class other_ananlysis(NodeAnalysis):
            dependencies = (list_modules_keys, )
        
        assert list_modules_keys.isInterModules()
        assert other_ananlysis.isInterModules()

        assert pm.gather(list_modules_keys, node1) == ['test', 'test2']
        assert pm.gather(list_modules_keys, node2) == ['test', 'test2']

    def test_dynamic_dependencies(self):
        normal = has_dynamic_dependencies(project_wide=False)
        project = has_dynamic_dependencies(project_wide=True)

        assert normal.dependencies != project.dependencies
        assert not normal.isInterModules()
        assert project.isInterModules()

    def test_simple_module_analysis(self):
        src = ('class A(object): ...\n'
               'class B: ...')
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        n = pm.gather(class_count, pm.modules['test'].node)
        assert n == 2
    
    def test_simple_function_analysis(self):
        src = 'def f(a:int, b:object=None, *, key:Callable, **kwargs):...'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        args = pm.gather(function_arguments, pm.modules['test'].node.body[0])
        assert [a.node.arg for a in args] == ['a', 'b', 'key', 'kwargs']
    
    def test_simple_class_analysis(self):
        src = 'class A(object, stuff): ...'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        bases = pm.gather(class_bases, pm.modules['test'].node.body[0])
        assert bases == ['object', 'stuff']
    
    def test_simple_node_analysis(self):
        src = 'v: list | set = None'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        nodes = pm.gather(node_list, pm.modules['test'].node.body[0])
        assert len([n for n in nodes if isinstance(n, ast.Name)]) == 3

    def test_simple_transformation(self):
        src = 'v = True'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        updates, node = pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert updates
        assert ast.unparse(node) == 'v = 1'

        # check it's not marked as updatesd when it's not.
        src = 'v = False'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        updates, node = pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert not updates
        assert ast.unparse(node) == 'v = False'
    
    def test_analysis_with_analysis_dependencies(self):
        src = ('def f(a, b=None, *, key, **kwargs):...\n'
               'def g(a, b=None, *, key, ):...')
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        r = pm.gather(function_accepts_any_keywords, pm.modules['test'].node.body[0])
        assert r is True

        r = pm.gather(function_accepts_any_keywords, pm.modules['test'].node.body[1])
        assert r is False
    
    def test_analysis_with_transforms_dependencies(self):
        src = 'v = True\nn = 1'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        n = pm.gather(literal_ones_count, pm.modules['test'].node)
        assert n == 2
    
    def test_analysis_with_transitive_transforms_dependencies_applies_still_eagerly(self):
        src = 'v = True\nn = 1'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))

        class literal_1_count(NodeAnalysis[int]):
            dependencies = (literal_ones_count, ) 
            # list the pass which has the transform in it's dependency
            # but do not access it with self.literal_ones_count, instead do the logic again
            def doPass(self, node: ast.AST) -> int:
                return len([
                    n for n in ast.walk(node) if isinstance(n, ast.Constant) 
                    and n.value is not True and n.value == 1])

        n = pm.gather(literal_1_count, pm.modules['test'].node)
        assert n == 2
    
    def test_preserved_analysis(self):
        # TODO: Think of more test cases here.
        # This seems lite for now.
        src = 'v = True\nn = 1\nclass A: ...'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        pm.gather(class_count, pm.modules['test'].node)
        pm.gather(node_list, pm.modules['test'].node)
        assert pm.cache.get(class_count, pm.modules['test'].node)
        assert pm.cache.get(node_list, pm.modules['test'].node)
        
        pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert pm.cache.get(class_count, pm.modules['test'].node)
        assert not pm.cache.get(node_list, pm.modules['test'].node)
    
    def test_transformation_rec_updates(self):
        """
        Test the Transformation.recAddNode and Transformation.recRemoveNode methods
        """
        # Quicksort Python One-liner
        src = 'v=True; qsort = lambda L: [] if L==[] else qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])'
        src = '\n'.join(src for _ in range(80))
        
        with catchtime('parse') as stimer:
            modules = [Module(ast.parse(f'v = {i}; {src}'), f'test_{i}') for i in range(20)]
        

        with catchtime('not-opti') as not_opti:
            fromPasses(modules, [transform_trues_into_ones])
        
        with catchtime('opti') as opti:
            fromPasses(modules, [transform_trues_into_ones_opti])
        
        # Yes it's faster...
        assert not_opti.time > opti.time

        # check it's not marked as updatesd when it's not.
        src = 'v = False'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        updates, node = pm.apply(transform_trues_into_ones_opti, pm.modules['test'].node)
        assert not updates
        assert ast.unparse(node) == 'v = False'
    
    # def test_passmanger_merge(self):
    #     # Quicksort Python One-liner
    #     src = 'qsort = lambda L: [] if L==[] else qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])'
    #     src = '\n'.join(src for _ in range(10))
        
    #     with catchtime('parse') as stimer:
    #         modules = [Module(ast.parse(f'v = {i}; {src}'), f'test_{i}') for i in range(20)]
        
    #     from libstatic._lib.analyses import def_use_chains
    #     set1, set2 = modules[:10], modules[10:]
    #     assert len(set1) == len(set2) == 10
        
    #     with catchtime('process set1'):
    #         pm1 = fromPasses(set1, [def_use_chains])
        
    #     with catchtime('process set2'):
    #         pm2 = fromPasses(set2, [def_use_chains])

    #     pm = PassManager()
        
    #     with catchtime('merging into new'):
    #         pm._merge(pm1)
    #         pm._merge(pm2)

    #     with catchtime('get all analyses from cache') as cacheAccess:
    #         for m in pm.modules.values():
    #             pm.gather(def_use_chains, m.node)
        
    #     assert cacheAccess.time < 0.001
    #     assert len(pm.modules) == 20

    
    def test_cache_cleared_when_module_removed(self):
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        pm.add_module(Module(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(requires_modules, mod1.node)
        assert pm.cache.get(requires_modules, mod1.node)

        pm.remove_module(mod2)

        assert pm.cache.get(node_list, mod1.node)
        assert pm.cache.get(requires_modules, mod1.node) is None
    
    def test_cache_cleared_when_module_added(self):
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        pm.add_module(Module(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(requires_modules, mod1.node)
        cache = pm.cache
        assert cache.get(requires_modules, mod1.node)

        pm.add_module(Module(ast.parse('import test'), 'test3'))

        assert cache.get(node_list, mod1.node)
        assert cache.get(requires_modules, mod1.node) is None
    

    def test_preserved_analysis_inter_modules(self):
        pass
        # TODO: an analysis that depends on other modules should be 
        # cleared from the cache if it's not listed on the preservesAnalyses attribute.
    
    def test_preserved_analysis_abstract(self):
        pass
        # TODO: an analysis that misses required parameters can be listed in both dependencies and
        # preservesAnalyses lists. When it's listed in preservesAnalyses all subclasses with any
        # required parameters and all same optional parameters are also preserved.

    def test_analysis_with_parameters_get_invalidated_like_others(self):
        pass
        # TODO: simple case where a transform invalidates a parameterized analysis
        # also when the transform preverses some of the derived parameterized analysis, but not all of them.
    
    def test_Pass_like_classmethod(self):
        class has_optional_parameters(NodeAnalysis):
            # We can create an infinity of subclasses of this type since mult can be any ints
            optionalParameters = dict(filterkilled=True, mult=1)
        
        # Test the __eq__ function
        pattern_mult_eq_1 = has_optional_parameters.like(
            filterkilled=lambda v: True, mult=lambda v:v==1
        )
        assert has_optional_parameters == pattern_mult_eq_1
        assert has_optional_parameters(mult=1) == pattern_mult_eq_1
        assert has_optional_parameters(mult=1, filterkilled=False) == pattern_mult_eq_1
        assert has_optional_parameters(mult=2) != pattern_mult_eq_1

        pattern_any = has_optional_parameters.like(
            filterkilled=lambda v: True, mult=lambda v: True
        )

        assert has_optional_parameters == pattern_any
        assert has_optional_parameters(mult=1) == pattern_any
        assert has_optional_parameters(mult=1, filterkilled=False) == pattern_any
        assert has_optional_parameters(mult=2) == pattern_any

        # Test inside containers
        assert has_optional_parameters in (pattern_any,)
        assert has_optional_parameters(mult=2) not in [pattern_mult_eq_1, has_optional_parameters(mult=1)]
        assert has_optional_parameters(mult=2) in [pattern_mult_eq_1, has_optional_parameters(mult=1), pattern_any]
    

    def test_preserved_analysis_subclass_explosion_issue(self):
        
        class class_count_with_parameters(NodeAnalysis[int]):
            # We can create an infinity of subclasses of this type since mult can be any ints
            optionalParameters = dict(filterkilled=False, 
                                      mult=1)
            dependencies = (class_count, )
            
            def doPass(self, node: Any) -> int:
                clscount = self.class_count
                return clscount * self.mult
        
        # optimized version of 'transform_trues_into_ones'
        class t1(transform_trues_into_ones):
                                 # This will only preserves the default version of the 
                                 # analysis with filterkilled=True and mult=1.
            __name__ = 'transform_trues_into_ones'
            preservesAnalyses = (class_count, class_count_with_parameters, )

        # better optimized version of 'transform_trues_into_ones'
        class t2(transform_trues_into_ones):
                                 # This will preserves all versions of the analysis
            __name__ = 'transform_trues_into_ones'
            preservesAnalyses = (class_count, class_count_with_parameters.like(filterkilled=lambda v:True, 
                                                                     mult=lambda v: True), )
        
        # better optimized version of 'transform_trues_into_ones'
        class t3(transform_trues_into_ones):
                                 # This will only preserves versions of the 
                                 # analysis with filterkilled=False and mult>0
            __name__ = 'transform_trues_into_ones'
            preservesAnalyses = (class_count, class_count_with_parameters.like(filterkilled=lambda v:not v, 
                                                                     mult=lambda v: v>0), )

        src = 'v = True\nn = 1\nclass A: ...\nclass B: ...'
        pm = PassManager()
        mod = ast.parse(src)
        pm.add_module(Module(
            mod, 'test', 'test.py', code=src, 
        ))

        def gather_analyses():
            pm.gather(class_count_with_parameters, mod)
            pm.gather(class_count_with_parameters.bind(mult=-4), mod)
            pm.gather(class_count_with_parameters.bind(mult=0), mod)
            pm.gather(class_count_with_parameters.bind(mult=3), mod)
            pm.gather(class_count_with_parameters.bind(mult=-4, filterkilled=True), mod)
            pm.gather(class_count_with_parameters.bind(mult=0, filterkilled=True), mod)
            pm.gather(class_count_with_parameters.bind(mult=3, filterkilled=True), mod)
            assert list(pm.cache.analyses()) == [
                class_count,
                class_count_with_parameters, 
                class_count_with_parameters.bind(mult=-4), 
                class_count_with_parameters.bind(mult=0), 
                class_count_with_parameters.bind(mult=3), 
                class_count_with_parameters.bind(mult=-4, filterkilled=True), 
                class_count_with_parameters.bind(mult=0, filterkilled=True), 
                class_count_with_parameters.bind(mult=3, filterkilled=True), 
            ]

        gather_analyses()
        
        # when the non-optimized version of the transformation is run, all analyses are invalidated
        # and needs to be recomputed.
        pm.apply(transform_trues_into_ones, mod)
        assert list(pm.cache.analyses()) == [
                    class_count,
            ]
        
        pm = PassManager()
        mod = ast.parse(src)
        pm.add_module(Module(
            mod, 'test', 'test.py', code=src, 
        ))
        gather_analyses()
        
        # Now let's apply t1 which defined the preserved analysis with only the default values.
        pm.apply(t1, mod)
        assert list(pm.cache.analyses()) == [
                    class_count,
                    class_count_with_parameters, 
            ]

        pm = PassManager()
        mod = ast.parse(src)
        pm.add_module(Module(
            mod, 'test', 'test.py', code=src, 
        ))
        gather_analyses()

        # Now t2, which preserves all versions of the analysis
        pm.apply(t2, mod)
        assert list(pm.cache.analyses()) == [
                    class_count,
                    class_count_with_parameters, 
                    class_count_with_parameters.bind(mult=-4), 
                    class_count_with_parameters.bind(mult=0), 
                    class_count_with_parameters.bind(mult=3), 
                    class_count_with_parameters.bind(mult=-4, filterkilled=True), 
                    class_count_with_parameters.bind(mult=0, filterkilled=True), 
                    class_count_with_parameters.bind(mult=3, filterkilled=True), 
            ]
        
        pm = PassManager()
        mod = ast.parse(src)
        pm.add_module(Module(
            mod, 'test', 'test.py', code=src, 
        ))
        gather_analyses()

        # Now t3, which preserves for filterkilled=False and mult>0
        pm.apply(t3, mod)
        assert list(pm.cache.analyses()) == [
                    class_count,
                    class_count_with_parameters, 
                    class_count_with_parameters.bind(mult=3), 
            ]
            

    def test_pass_instrumentation_run_times(self):
        pass # TODO: We should be able to hack something to get the run times of all analyses
        # 

    def test_pass_can_gather_analyses_not_listed_in_dependencies(self):

        class test_analysis(FunctionAnalysis):
            dependencies = (node_list, )
            def doPass(self, node: ast.AST) -> Any:
                self.node_list # access it, why? idk..
                return self.passmanager.gather(function_accepts_any_keywords, node) # ok even not in depedencies
        
        src = 'def f(a, *, b, **k):pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        assert pm.gather(test_analysis, pm.modules['test'].node.body[0]) is True
    
    def test_pass_can_apply_transformation_not_listed_in_dependencies(self):
        ...

    def test_not_using_modules_analysis_cannot_gather_other_using_modules_analysis(self):
        # If none of the statically declared dependencies depends on the 'modules'
        # analsis; trying to gather an inter-modules analysis manually with self.passmanager.gather(analysis, node)
        # will fail.
        class main_intra_module_analysis(NodeAnalysis):
            dependencies = (node_list, )
            def doPass(self, node: ast.AST) -> Any:
                self.passmanager.gather(has_dynamic_dependencies(project_wide=True), node) # raises
        
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        with self.assertRaises(TypeError):
            pm.gather(main_intra_module_analysis, pm.modules['test'].node)
                
    def test_not_using_modules_analysis_cannot_apply_transformation_using_modules_analysis(self):
        pass # TODO: same for a transformation

    def test_analysis_that_applies_a_transformation(self):
        # using self.passmanager.apply from the doPass method.
        pass

    def test_analysis_that_applies_a_transformation_to_another_module(self):
        # using self.passmanager.apply from the doPass method.
        pass
    
    def test_analysis_valid_dependencies_order(self):
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        
        pm.gather(still_valid_analysis_dependencies_order, pm.modules['test'].node)
    
    def test_analysis_invalid_no_doPass_method(self):
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        with self.assertRaises(Exception):
            pm.gather(invalid_analysis_no_doPass_method, pm.modules['test'].node)

    def test_analysis_not_run_if_not_accessed(self):
        # when an analysis is declared as a dependency a descriptor is used to
        # run the analysis only when accessed with self.analysis_name
        ...
        class dependency_that_raises_an_error(NodeAnalysis[None]):
            def doPass(self, node: ast.AST) -> None:
                raise RuntimeError()
        
        class dependent(NodeAnalysis[None]):
            dependencies = (dependency_that_raises_an_error, )
            optionalParameters = {'access_dependency_that_raises_an_error': False}
            def doPass(self, node: ast.AST) -> None:
                if self.access_dependency_that_raises_an_error:
                    self.dependency_that_raises_an_error
        
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))

        pm.gather(dependent, pm.modules['test'].node)

        with self.assertRaises(RuntimeError):
            pm.gather(dependent(access_dependency_that_raises_an_error=True), pm.modules['test'].node)

    def test_transformation_with_analysis_dependencies(self):
        # after the transformation have been executed, if the dependent analysis is not
        # in the preservesAnalyses collection, it it revomed from the cache
        ...
    
    def test_transformation_with_unsuported_cyclic_dependencies(self):
        ...
        # TODO: Thid would br a transformation that transitively depends on itselft. 
    
    def test_transformation_with_suported_cyclic_dependencies(self):
        # this uses a feeature that is still not implemented... 
        # when a pass needs itself for the same node again it can either fail
        # or provide a fallcack function that will return a dummy object or in the 
        # best case do some addition logic to sort it out.
        ...
    
    def test_analysis_with_unsuported_cyclic_dependencies(self):
        # so this one will use the simple goto definition that goes in circle: simple.
        ...
    
    def test_analysis_with_suported_cyclic_dependencies(self):
        # the type inference analysis typically can be recursive, so we'll use a simple 
        # version of type inference that will always return the unknown type when it recurses.
        ...
    
    def test_function_to_module_analysis_promotion(self):
        # we'll gather the results of whether the function accepts any keywords on a whole module
        node = ast.parse(dedent('''
        def f1(a, *, b):...                   
        def f2(**kw):
            def f3(*, c, d):...
            class s:
                def __init__(self, ):...
        '''))
        class test_analysis(ModuleAnalysis[None]):
            requiredParameters = ('expected_number_of_functions', )
            dependencies = (function_arguments.proxy(), )
            def doPass(self, node: ast.Module) -> True:
                assert isinstance(self.function_arguments, passmanager.GetProxy)
                functions = tuple(f for f in ast.walk(node) if isinstance(f, ast.FunctionDef))
                assert len(functions) == self.expected_number_of_functions
                assert all(self.function_arguments.get(f) for f in functions)
                return True
        
        class remove_all_init_methods(Transformation):
            """Dummy transform that removed all __init__ methods."""
            dependencies = (analyses.node_enclosing_scope, )
            def doPass(self, node: ast.Module | None) -> ast.Module:
                updates = False
                class transformer(ast.NodeTransformer):
                    def visit_FunctionDef(tself, node: ast.FunctionDef) -> Any:
                        if node.name == '__init__' and isinstance(
                            self.passmanager.gather(analyses.node_enclosing_scope, node), ast.ClassDef):
                            nonlocal updates
                            updates = True
                            return None
                        return tself.generic_visit(node)
                transformer().visit(node)
                self.update = updates
                return node

        pm = PassManager()
        pm.add_module(Module(node, 'test'))
        assert pm.gather(test_analysis(expected_number_of_functions=4), node)
        assert list(pm.cache.analyses()) == [function_arguments, test_analysis(expected_number_of_functions=4),]

        pm.apply(remove_all_init_methods, node)
        assert list(pm.cache.analyses()) == []
        
        assert pm.gather(test_analysis(expected_number_of_functions=3), node)
        assert list(pm.cache.analyses()) == [function_arguments, test_analysis(expected_number_of_functions=3),]
        
        pm.apply(remove_all_init_methods, node) # it did not updated
        assert list(pm.cache.analyses()) == [function_arguments, test_analysis(expected_number_of_functions=3),]

        v = pm.gather(function_arguments.proxy(), node)
        
    
    def test_class_to_module_analysis_promotion(self):
        # class bases on a whole module
        ...

    def test_do_not_cache_analysis_honored(self):
        # TODO: It seem that _AnalysisProxy types are still added to the caches
        ...
