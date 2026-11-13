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
from libstatic.passmanager3 import (PassManager, Tree, Forest, analysis, transformation, 
                                    PassLike, PassPrototype, PassInstance, Connector,
                                    _TRANSFORMATION, _ANALYSIS, _FOREST, _TREE, _NODE)
from libstatic import passes3

import beniget

# passmanager test factory

def fromPasses(modules: Collection[Tree], passes: Collection[PassLike]) -> PassManager:
    pm = PassManager()
    _transforms, _analyses = (), ()
    
    for p in passes:
        assert p.proto.runs_on in (_TREE, _NODE)
        if p.proto.kind is _TRANSFORMATION:
            _transforms += (p,)
        elif p.proto.kind is _ANALYSIS:
            _analyses += (p,)
        else: assert False

    for m in modules:
        pm.add(m)
        for t in _transforms:
            pm.apply(t, m)
        for a in _analyses:
            pm.gather(a, m)
    
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


############### test passes

class _node_list(ast.NodeVisitor):
    "List of all children in the node recursively depth first"
    def visit(self, node: ast.AST) -> None:
        self.result.append(node)
        super().visit(node)

    def gather(self, node: ast.AST) -> list[ast.AST]:
        self.result = []
        self.visit(node)
        return self.result

@analysis(on=ast.AST)
def node_list(_, node): 
    """
    List of all children in the node recursively depth first

    @rtype: list[AST]
    """
    yield 'result', _node_list().gather(node)

@analysis(on=ast.ClassDef)
def calss_bases(_, node: ast.ClassDef):
    """
    The bases of a class as strings

    @rtype: list[str]
    """
    yield 'result', [ast.unparse(n) for n in node.bases]

@analysis(on=(ast.FunctionDef, ast.AsyncFunctionDef))
def function_arguments(_, node: ast.FunctionDef | ast.AsyncFunctionDef):
    """
    List of function arguments

    @rtype: list[ArgSpec]
    """
    yield 'result', list(iter_arguments(node.args))

@analysis(on=(ast.FunctionDef, ast.AsyncFunctionDef), 
          dependencies=[function_arguments])
def function_accepts_any_keywords(c, node: ast.FunctionDef | ast.AsyncFunctionDef):
    """
    Whether a function will accept any keyword argument, based on it's signature

    @rtype: bool
    """
    yield 'result', any(a.kind == inspect.Parameter.VAR_KEYWORD for a in c.deps.function_arguments)

class _class_count(LocalStmtVisitor):
    """
    Counts the number of classes in the locals of the given node.
    """
    
    def visit_ClassDef(self, node):
        self.result += 1
    
    def gather(self, node: ast.AST) -> int:
        self.result = 0
        self.visit(node)
        return self.result

@analysis(on=ast.AST)
def class_count(_, node): 
    """
    @rtype: list[AST]
    """
    yield 'result', _class_count().gather(node)

class _simple_symbol_table(ast.NodeVisitor):
    "Builds a symbol table, for the module level only"
    
    def gather(self, node: ast.Module) -> dict[str, list[ast.AST]]:
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

@analysis(on=Tree)
def simple_symbol_table(_, node: Tree):
    """
    @rtype: dict[str, list[ast.AST]]
    """
    yield 'result', _simple_symbol_table().gather(node.root)

@analysis(on=ast.Name, dependencies=(simple_symbol_table, passes3.ancestors))
def simple_goto_def(c, node):
    """
    goto the definition of th symbol, but only works at module level.
    
    @rtype: ast.AST | None
    """
    if node.ctx.__class__.__name__ == 'Store':
        try:
            assign = next(n for n in c.deps.ancestors.parents(node) 
                            if n.__class__.__name__ in ('Assign', 'AnnAssign'))
            return get_stored_value(node, assign)
        except Exception as e:
            if __debug__:
                print(str(e), file=sys.stderr)
            yield 'result', None
    
    elif node.ctx.__class__.__name__ == 'Load':
        defi = c.deps.simple_symbol_table[node.id]
        if defi: 
            yield 'result', defi[-1]
        yield 'result', None
    
    else:
        raise TypeError(node.ctx)

class _transform_trues_into_ones(ast.NodeTransformer):
    "True -> 1"
    update = False

    def visit_Constant(self, node):
        if node.value is True:
            self.update = True
            return ast.Constant(value=1)
        else:
            return node
    
    def apply(self, node: ast.Module):
        self.visit(node)
        return self.update

@transformation(on=Tree)
def transform_trues_into_ones(_, tree):
    yield 'update', _transform_trues_into_ones().apply(tree)
    yield 'preserved', [class_count]

@transformation(on=Tree, dependencies=[passes3.ancestors])
def transform_trues_into_ones_opti(c, tree):

    class _transform_trues_into_ones_opti(ast.NodeTransformer):
        "True -> 1"
        update = False
        
        def visit_Constant(self, node):
            if node.value is True:
                replacement = ast.Constant(value=1)
                self.update = True

                # update ancestors ~3 lines per replacement, 
                # to repeat for each analyses that we need to preserve...
                ans = c.deps.ancestors
                passes3.fix_ancestors(ans, node, replacement)
                passes3.prune_ancestors(ans, node)

                return replacement
            else:
                return node
        
        def apply(self, node: ast.AST) -> bool:
            self.visit(node)
            return self.update
    
    yield 'update', _transform_trues_into_ones_opti().apply(tree)
    yield 'preserved', [class_count, passes3.ancestors]

@analysis(on=ast.AST)
def has_optional_parameters(c, node, *, filter=False, inherited=False):
    yield 'result', 1

@analysis(on=ast.AST)
def has_required_parameters(c, node, *, name, thing):
    yield 'result', 1

@analysis(on=ast.AST)
def has_both_required_and_optional_parameters(c, node, *, name, thing, filter=False, inherited=False):
    yield 'result', 1

@analysis(on=Forest)
def runs_on_forest(c, node: Forest):
    assert len(node) >= 1
    yield 'result', 1

@analysis(on=ast.AST, 
          dependencies = (has_required_parameters, runs_on_forest))
def has_dynamic_dependencies(c, _, *, project_wide=False):    
    if project_wide:
        c.deps.has_required_parameters
        assert c.deps.runs_on_forest == 1
    
    yield 'result', 1

class _OneCount(ast.NodeVisitor):
    def visit_Constant(self, node):
        if node.value is not True and node.value == 1:
            self.result += 1

    def gather(self, node: ast.AST) -> int:
        self.result = 0
        self.visit(node)
        return self.result

@analysis(on=Tree, dependencies = (transform_trues_into_ones, 
                    # this analysis is not used, yes I know
                    class_count) )
def literal_ones_count(_, node):
    "counts the number of literal '1'"
    yield 'result', _OneCount().gather(node)

@analysis(on=Tree, dependencies = (node_list, transform_trues_into_ones))
def still_valid_analysis_dependencies_order(_, node): 
    yield 'result', 1

@analysis(on=Tree)
def  invalid_analysis_not_returning_anything(_, node): pass

@analysis(on=Tree)
def  invalid_analysis_not_returning_required_keys(_, node): return {'foo':1}

@analysis(on=Tree)
def  invalid_transformation_not_returning_required_keys(_, node): return {'foo':1}


############## test cases begins

class TestTestAnalysis(TestCase):
    
    def test_simple_symbol_table(self):
        src = ('class A(object): ...\n'
               'def f():...\n'
               'var = x = True\n'
               'from x import *\n'
               'import pydoctor.driver\n')
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', filename='test.py', code=src, 
        ))
        symtable = pm.gather(simple_symbol_table, 'test')
        assert len(symtable) == 6
        assert list(symtable) == ['A', 'f', 'var', 'x', '*', 'pydoctor']
    
    def test_simple_goto_def(self):
        src = ('from x import y\n'
               'var = y\n'
               'var')
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', filename='test.py', code=src, 
        ))
        varuse = pm.trees['test'].root.body[-1].value
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
    

class TestPassManagerAncestorsAnalysis(TestCase):
    """
    This test case ensures that the ancestors are recomputed when a transformation is applied OR
    fixed when an opimized transformation is applied.
    """

    def test_ancestors_recomputed(self):
        src = 'v = True'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', filename='test.py', code=src, 
        ))
        # that node is going to be removed from the tree by the transform
        True_node = pm.trees['test'].root.body[0].value
        # gather the ancestors
        ans = pm.gather(passes3.ancestors, 'test')
        # the analysis result is cached
        assert len(pm.cache.allkeys()) == 1
        assert True_node in ans._parents
        # the same instance is returned because it's cached
        assert ans is pm.gather(passes3.ancestors, 'test')
        # the transformation did updated the ast
        assert pm.apply(transform_trues_into_ones, 'test')
        # the cahed analysis was cleared
        assert list(pm.cache.allkeys()) == []
        # a new instance is returned
        assert ans is not pm.gather(passes3.ancestors, 'test')

        ans = pm.gather(passes3.ancestors, 'test')
        # that does not contains our removed node
        assert True_node not in ans._parents

        One_node = pm.trees['test'].root.body[0].value
        assert One_node in ans._parents

        # when we remove the module, it's analyses are cleared from cache
        pm.remove(pm.trees['test'])
        assert list(pm.cache.allkeys()) == []

    
    def test_ancestors_fixed(self):
        src = 'v = True'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', filename='test.py', code=src, 
        ))
        # that node is going to be removed from the tree by the transform
        True_node = pm.trees['test'].root.body[0].value
        # gather the ancestors
        ans = pm.gather(passes3.ancestors, 'test')
        # the analysis result is cached
        assert len(pm.cache.allkeys()) == 1
        assert True_node in ans._parents
        # the same instance is returned because it's cached
        assert ans is pm.gather(passes3.ancestors, 'test')
        # the transformation did updated the ast
        assert pm.apply(transform_trues_into_ones_opti, 'test')
        # the cahed analysis was preserved!!!
        assert len(pm.cache.allkeys()) == 1
        # the instance is returned!!!
        assert ans is pm.gather(passes3.ancestors, 'test')

        ans = pm.gather(passes3.ancestors, 'test')
        # that does not contains our removed node, because it was pruned
        assert True_node not in ans._parents

        One_node = pm.trees['test'].root.body[0].value
        assert One_node in ans._parents

        # when we remove the module, it's analyses are cleared from cache
        pm.remove(pm.trees['test'])
        assert list(pm.cache.allkeys()) == []


class TestPassManagerFramework(TestCase):
    
    def test_analysis_class_call_optional_parameters(self):
        # when calling a class with already the arguments default, this has not affect
        assert has_optional_parameters() is has_optional_parameters(filter=False)
        assert has_optional_parameters() is has_optional_parameters(filter=False, inherited=False)
        
        # this creates a new prototypes
        assert has_optional_parameters() is not has_optional_parameters(filter=True)
        assert has_optional_parameters(filter=True) is has_optional_parameters(filter=True)
        assert has_optional_parameters(filter=True) is has_optional_parameters(filter=True, inherited=False)
        
        pm = PassManager()
        pm.add(Tree(ast.parse('pass'), 'test'))
        tree = pm.trees['test']
        root = tree.root
        identifier = tree.identifier

        # these NODE analyses can be run on:
        # - tree [, node]
        # - root [, node]
        # - identifier [, node]
        
        assert pm.gather(has_optional_parameters, tree) == pm.gather(has_optional_parameters, root) == pm.gather(has_optional_parameters, identifier)
        assert pm.gather(has_optional_parameters(filter=True), tree) == pm.gather(has_optional_parameters(filter=True), root) == pm.gather(has_optional_parameters(filter=True), identifier)
        assert pm.gather(has_optional_parameters(filter=True, inherited=True), tree) == pm.gather(has_optional_parameters(filter=True, inherited=True), root) == pm.gather(has_optional_parameters(filter=True, inherited=True), identifier)
        
        assert pm.gather(has_optional_parameters, tree, root) == pm.gather(has_optional_parameters, root, root) == pm.gather(has_optional_parameters, identifier, root)
        assert pm.gather(has_optional_parameters(filter=True), tree, root) == pm.gather(has_optional_parameters(filter=True), root, root) == pm.gather(has_optional_parameters(filter=True), identifier, root)
        assert pm.gather(has_optional_parameters(filter=True, inherited=True), tree, root) == pm.gather(has_optional_parameters(filter=True, inherited=True), root, root) == pm.gather(has_optional_parameters(filter=True, inherited=True), identifier, root)

    def test_analysis_class_call_multiple_calls(self):
        assert has_optional_parameters(filter=True)(filter=False) is has_optional_parameters(filter=False)
        assert has_both_required_and_optional_parameters(name='show')(filter=False) is has_both_required_and_optional_parameters(name='show', filter=False)
        
    def test_analysis_class_call_required_parameters(self):

        pm = PassManager()
        pm.add(Tree(ast.parse('pass'), 'test'))
        node = pm.trees['test'].root

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

    def test_tree_knowkedge_analysis(self):
        
        @analysis(on=Forest)
        def known_tree_identifiers(_, forest: Forest):
            yield 'result', {t.identifier for t in forest}
        
        @analysis(on=ast.Name)
        def is_known_module_name(c:Connector, node: ast.Name):
            yield 'result', node.id in c.gather(known_tree_identifiers)
        
        pm = PassManager()
        pm.add(Tree(ast.parse('pass'), 'test'))
        pm.add(Tree(ast.parse('import test'), 'test2'))

        test_name = ast.Name(id='test', ctx=ast.Load())
        other_name = ast.Name(id='other', ctx=ast.Load())

        completedpass = pm.run(is_known_module_name, 'test', test_name)
        assert completedpass.result is True
        assert completedpass.knowledge == _FOREST

        completedpass = pm.run(is_known_module_name, 'test', other_name)
        assert completedpass.result is False
        assert completedpass.knowledge == _FOREST

    def test_dynamic_dependencies(self):
        normal = has_dynamic_dependencies(project_wide=False)
        project = has_dynamic_dependencies(project_wide=True)
        
        pm = PassManager()
        pm.add(Tree((mod:=ast.parse('pass')), 'test'))

        pm.gather(normal, mod)

    
    def test_simple_module_analysis(self):
        src = ('class A(object): ...\n'
               'class B: ...')
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        n = pm.gather(class_count, pm.modules['test'].node)
        assert n == 2
    
    def test_simple_function_analysis(self):
        src = 'def f(a:int, b:object=None, *, key:Callable, **kwargs):...'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        args = pm.gather(function_arguments, pm.modules['test'].node.body[0])
        assert [a.node.arg for a in args] == ['a', 'b', 'key', 'kwargs']
    
    def test_simple_class_analysis(self):
        src = 'class A(object, stuff): ...'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        bases = pm.gather(class_bases, pm.modules['test'].node.body[0])
        assert bases == ['object', 'stuff']
    
    def test_simple_node_analysis(self):
        src = 'v: list | set = None'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        nodes = pm.gather(node_list, pm.modules['test'].node.body[0])
        assert len([n for n in nodes if isinstance(n, ast.Name)]) == 3

    def test_simple_transformation(self):
        src = 'v = True'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        updates, node = pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert updates
        assert ast.unparse(node) == 'v = 1'

        # check it's not marked as updatesd when it's not.
        src = 'v = False'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        updates, node = pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert not updates
        assert ast.unparse(node) == 'v = False'
    
    def test_analysis_with_analysis_dependencies(self):
        src = ('def f(a, b=None, *, key, **kwargs):...\n'
               'def g(a, b=None, *, key, ):...')
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        r = pm.gather(function_accepts_any_keywords, pm.modules['test'].node.body[0])
        assert r is True

        r = pm.gather(function_accepts_any_keywords, pm.modules['test'].node.body[1])
        assert r is False
    
    def test_analysis_with_transforms_dependencies(self):
        src = 'v = True\nn = 1'
        pm = PassManager()
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        n = pm.gather(literal_ones_count, pm.modules['test'].node)
        assert n == 2
    
    def test_analysis_with_transitive_transforms_dependencies_applies_still_eagerly(self):
        src = 'v = True\nn = 1'
        pm = PassManager()
        pm.add(Tree(
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
        pm.add(Tree(
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
            modules = [Tree(ast.parse(f'v = {i}; {src}'), f'test_{i}') for i in range(20)]
        

        with catchtime('not-opti') as not_opti:
            fromPasses(modules, [transform_trues_into_ones])
        
        with catchtime('opti') as opti:
            fromPasses(modules, [transform_trues_into_ones_opti])
        
        # Yes it's faster...
        assert not_opti.time > opti.time

        # check it's not marked as updatesd when it's not.
        src = 'v = False'
        pm = PassManager()
        pm.add(Tree(
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
    #         modules = [Tree(ast.parse(f'v = {i}; {src}'), f'test_{i}') for i in range(20)]
        
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
        pm.add(Tree(ast.parse('pass'), 'test'))
        pm.add(Tree(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(runs_on_forest, mod1.node)
        assert pm.cache.get(runs_on_forest, mod1.node)

        pm.remove_module(mod2)

        assert pm.cache.get(node_list, mod1.node)
        assert pm.cache.get(runs_on_forest, mod1.node) is None
    
    def test_cache_cleared_when_module_added(self):
        pm = PassManager()
        pm.add(Tree(ast.parse('pass'), 'test'))
        pm.add(Tree(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(runs_on_forest, mod1.node)
        cache = pm.cache
        assert cache.get(runs_on_forest, mod1.node)

        pm.add(Tree(ast.parse('import test'), 'test3'))

        assert cache.get(node_list, mod1.node)
        assert cache.get(runs_on_forest, mod1.node) is None
    

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
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        
        pm.gather(still_valid_analysis_dependencies_order, pm.modules['test'].node)
    
    def test_analysis_invalid_no_doPass_method(self):
        src = 'pass'
        pm = PassManager()
        pm.add(Tree(
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
        pm.add(Tree(
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
        pm.add(Tree(node, 'test'))
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
    
    def test_isComplete(self):
        ...
