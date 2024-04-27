from __future__ import annotations

from collections import defaultdict
import sys
from textwrap import dedent
from typing import Any, Mapping

from unittest import TestCase

import ast, inspect

# implementation details
from libstatic._lib.arguments import ArgSpec, iter_arguments # Yields all arguments of the given ast.arguments node as ArgSpec instances.
from libstatic._lib.assignment import get_stored_value # Given an ast.Name instance with Store context and it's parent assignment statement, figure out the right hand side expression that is stored in the symbol.

# main framework module we're testing
from libstatic._lib.passmanager import (PassManager, Module, NodeAnalysis, FunctionAnalysis, 
                                        ClassAnalysis, ModuleAnalysis, Transformation)
from libstatic._lib import passmanager, analyses

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

# assert node_list(v=1) is node_list(v=1)
# assert node_list(v=1).v == 1

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

class class_count(ModuleAnalysis[int], ast.NodeVisitor):
    """
    Counts the number of classes in the module
    """
    
    def visit_ClassDef(self, node):
        self.result += 1
        for n in node.body:
            self.visit(n)
    
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

class simple_goto_def(NodeAnalysis[ast.AST | None]):
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
    preserves_analysis = (class_count, )

    def visit_Constant(self, node):
        if node.value is True:
            self.update = True
            return ast.Constant(value=1)
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
        
        assert list_modules_keys._usesModulesTransitive()
        assert other_ananlysis._usesModulesTransitive()

        assert pm.gather(list_modules_keys, node1) == ['test', 'test2']
        assert pm.gather(list_modules_keys, node2) == ['test', 'test2']

    def test_dynamic_dependencies(self):
        normal = has_dynamic_dependencies(project_wide=False)
        project = has_dynamic_dependencies(project_wide=True)

        assert normal.dependencies != project.dependencies
        assert not normal._usesModulesTransitive()
        assert project._usesModulesTransitive()

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
        assert pm._passmanagers[pm.modules['test']].cache.get(class_count, pm.modules['test'].node)
        assert pm._passmanagers[pm.modules['test']].cache.get(node_list, pm.modules['test'].node)
        
        pm.apply(transform_trues_into_ones, pm.modules['test'].node)
        assert pm._passmanagers[pm.modules['test']].cache.get(class_count, pm.modules['test'].node)
        assert not pm._passmanagers[pm.modules['test']].cache.get(node_list, pm.modules['test'].node)
    
    
    def test_cache_cleared_when_module_removed(self):
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        pm.add_module(Module(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(requires_modules, mod1.node)
        mod1cache = pm._passmanagers[mod1].cache
        assert mod1cache.get(requires_modules, mod1.node)

        pm.remove_module(mod2)

        assert mod1cache.get(node_list, mod1.node)
        assert mod1cache.get(requires_modules, mod1.node) is None
    
    def test_cache_cleared_when_module_added(self):
        pm = PassManager()
        pm.add_module(Module(ast.parse('pass'), 'test'))
        pm.add_module(Module(ast.parse('import test'), 'test2'))
        mod1 = pm.modules['test']
        mod2 = pm.modules['test2']

        pm.gather(node_list, mod1.node)
        pm.gather(requires_modules, mod1.node)
        mod1cache = pm._passmanagers[mod1].cache
        assert mod1cache.get(requires_modules, mod1.node)

        pm.add_module(Module(ast.parse('import test'), 'test3'))

        assert mod1cache.get(node_list, mod1.node)
        assert mod1cache.get(requires_modules, mod1.node) is None
    

    def test_preserved_analysis_inter_modules(self):
        pass
        # TODO: an analysis that depends on other modules should be 
        # cleared from the cache if it's not listed on the preserves_analysis attribute.
    
    def test_preserved_analysis_abstract(self):
        pass
        # TODO: an analysis that misses required parameters can be listed in both dependencies and
        # preserves_analysis lists. When it's listed in preserves_analysis all subclasses with any
        # required parameters and all same optional parameters are also preserved.

    def test_analysis_with_parameters_get_invalidated_like_others(self):
        pass
        # TODO: simple case where a transform invalidates a parameterized analysis
        # also when the transform preverses some of the derived parameterized analysis, but not all of them.
    
    def test_preserved_analysis_subclass_explosion_issue(self):
        pass
        # TODO: so the issue araise when dealing with more than two optional parameters. 
        # it's required for the transformation to list all variant with non-default parameters
        # that are preserved by the transform.
        # To remidiate this situation we can:
        # -  give-up on the class-call subclassing feature and move the analyses parameter in constructor
        #    this means that we should add the parameters to the cache keys. the thing is the dependencies
        #    currently needs to be declared at the class level, so in the case of dynamic dependencies this
        #    does not work. 
        # -  not a real solution by maybe move from using a metaclass and use __class_getitem__ ?
        #    which would result into using analysis_name[keyword=42]
        #  - a real solution would be to generate an ensemble of subclass that we can match against existing 
        #    analyses in the cache (this is what it is about afer all - this plus the fact that dependencies must be static
        #    at class level - but this might also be the thing to reconsider...)
        #    A solution would be to overwrite __invert__ or something such that we can create a "ensemble of analyses" like
        #    analysis_name.like(keyword=lambda v: bool(v), other_unimportant=lambda v:True)
        #    or shorter: analysis_name.like(keyword=True, other_unimportant=None) # all keywords must be given when creating a ensemble.
        #    this would be used in the Transformation.preserves_analysis attribute to indicate the transform preserves
        #    a varieties of derivation of the given analysis.
    
    def test_pass_instrumentation_run_times(self):
        pass # TODO: We should be able to hack something to get the run times of all analyses
        # 

    def test_not_using_modules_analysis_cannot_gather_other_using_modules_analysis(self):
        pass # TODO: If none of the statically declared dependencies depends on the 'modules'
        # analsis; trying to gather an analysis manually with self.passmanager.gather(analysis, node)
        # will fail.

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

    def test_transformation_with_analysis_dependencies(self):
        # after the transformation have been executed, if the dependent analysis is not
        # in the preserves_analysis collection, it it revomed from the cache
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
        ...
    
    def test_class_to_module_analysis_promotion(self):
        # class bases on a whole module
        ...

