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
from libstatic._lib.imports import ParseImportedNames, ImportInfo
from libstatic._lib.ivars import _compute_ivars
from libstatic._lib import exceptions
# main framework module we're testing
from libstatic._lib.passmanager import (PassManager, Module, NodeAnalysis, FunctionAnalysis, 
                                        ClassAnalysis, ModuleAnalysis, Transformation, walk)
from libstatic._lib import passmanager

from beniget.standard import DefUseChains, UseDefChains
import beniget

# test analysis

class node_list(NodeAnalysis[list[ast.AST]], ast.NodeVisitor):
    "List of all children in the node recursively depth first"
    def visit(self, node: ast.AST) -> list[ast.AST]:
        self.result.append(node)
        super().visit(node)
        return self.result

    def do_pass(self, node: ast.AST) -> list[ast.AST]:
        self.result = []
        self.visit(node)
        return self.result

# assert node_list(v=1) is node_list(v=1)
# assert node_list(v=1).v == 1

class class_bases(ClassAnalysis[list[str]]):
    "The bases of a class as strings"
    def do_pass(self, node: ast.ClassDef) -> list[str]:
        return [ast.unparse(n) for n in node.bases]

class function_arguments(FunctionAnalysis[list[ArgSpec]]):
    "List of function arguments"
    def do_pass(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ArgSpec]:
        return list(iter_arguments(node.args))

class function_accepts_any_keywords(FunctionAnalysis[bool]):
    "Whether a function will accept any keyword argument, based on it's signature"
    
    dependencies = (function_arguments, )
    
    def do_pass(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return any(a.kind == inspect.Parameter.VAR_KEYWORD for a in self.function_arguments)

class class_count(ModuleAnalysis[int], ast.NodeVisitor):
    """
    Counts the number of classes in the module
    """
    
    def visit_ClassDef(self, node):
        self.result += 1
        for n in node.body:
            self.visit(n)
    
    def do_pass(self, node: ast.Module) -> int:
        self.result = 0
        self.visit(node)
        return self.result

class ancestors(ModuleAnalysis[Mapping[ast.AST, list[ast.AST]]], ast.NodeVisitor):
    '''
    Associate each node with the list of its ancestors

    Based on the tree view of the AST: each node has the Module as parent.
    The result of this analysis is a dictionary with nodes as key,
    and list of nodes as values.

    >>> from pprint import pprint
    >>> mod = ast.parse('v = lambda x: x+1; w = 2')
    >>> pm = PassManager('t')
    >>> pprint({location(n):[location(p) for p in ps] for n,ps in pm.gather(ancestors, mod).items()})
    {'ast.Add at ?:?': ['ast.Module at ?:?',
                        'ast.Assign at ?:1',
                        'ast.Lambda at ?:1:4',
                        'ast.BinOp at ?:1:14'],
     'ast.Assign at ?:1': ['ast.Module at ?:?'],
     'ast.Assign at ?:1:19': ['ast.Module at ?:?'],
     'ast.BinOp at ?:1:14': ['ast.Module at ?:?',
                             'ast.Assign at ?:1',
                             'ast.Lambda at ?:1:4'],
     'ast.Constant at ?:1:16': ['ast.Module at ?:?',
                                'ast.Assign at ?:1',
                                'ast.Lambda at ?:1:4',
                                'ast.BinOp at ?:1:14'],
     'ast.Constant at ?:1:23': ['ast.Module at ?:?', 'ast.Assign at ?:1:19'],
     'ast.Lambda at ?:1:4': ['ast.Module at ?:?', 'ast.Assign at ?:1'],
     'ast.Load at ?:?': ['ast.Module at ?:?',
                         'ast.Assign at ?:1',
                         'ast.Lambda at ?:1:4',
                         'ast.BinOp at ?:1:14',
                         'ast.Name at ?:1:14'],
     'ast.Module at ?:?': [],
     'ast.Name at ?:1': ['ast.Module at ?:?', 'ast.Assign at ?:1'],
     'ast.Name at ?:1:11': ['ast.Module at ?:?',
                            'ast.Assign at ?:1',
                            'ast.Lambda at ?:1:4',
                            'ast.arguments at ?:?'],
     'ast.Name at ?:1:14': ['ast.Module at ?:?',
                            'ast.Assign at ?:1',
                            'ast.Lambda at ?:1:4',
                            'ast.BinOp at ?:1:14'],
     'ast.Name at ?:1:19': ['ast.Module at ?:?', 'ast.Assign at ?:1:19'],
     'ast.Param at ?:?': ['ast.Module at ?:?',
                          'ast.Assign at ?:1',
                          'ast.Lambda at ?:1:4',
                          'ast.arguments at ?:?',
                          'ast.Name at ?:1:11'],
     'ast.Store at ?:?': ['ast.Module at ?:?',
                          'ast.Assign at ?:1:19',
                          'ast.Name at ?:1:19'],
     'ast.arguments at ?:?': ['ast.Module at ?:?',
                              'ast.Assign at ?:1',
                              'ast.Lambda at ?:1:4']}
    '''

    def generic_visit(self, node):
        self.result[node] = current = self.current
        self.current += node,
        super().generic_visit(node)
        self.current = current

    visit = generic_visit

    def do_pass(self, node: ast.Module) -> dict[ast.AST, Module]:
        self.result = dict()
        self.current = tuple()
        self.visit(node)
        return self.result

class expand_expr(NodeAnalysis[str|None]):
    ...

class node_ancestor(NodeAnalysis[ast.AST]):
    """
    First node ancestor of class C{klass}
    """
    dependencies = (ancestors, )
    requiredParameters = ('klass', )

    def do_pass(self, node: ast.AST) -> ast.AST:
        # special case module access for speed.
        #             don't forget klass can be a tuple
        if isinstance(self.klass, type) and issubclass(self.klass, ast.Module):
            try:
                mod = next(iter(self.ancestors[node]))
            except StopIteration:
                pass
            else:
                if isinstance(mod, self.klass):
                    return mod  # type: ignore
        for n in reversed(self.ancestors[node]):
            # TODO: Use TypeGard annotation
            if isinstance(n, self.klass):
                return n  # type: ignore
        raise exceptions.StaticValueError(node, f"node has no parent of type {self.klass}")

class node_enclosing_scope(NodeAnalysis[ast.AST|None]):
    """
    Get the first enclosing scope of this use or definition.
    Returns None only of the definition is a Module.
    """
    dependencies = (node_ancestor(klass=(
            ast.SetComp,
            ast.DictComp,
            ast.ListComp,
            ast.GeneratorExp,
            ast.Lambda,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Module,
        )), )
    
    def do_pass(self, node: ast.AST) -> ast.AST | None:
        if isinstance(node, ast.Module):
            return None
        return self.node_ancestor

class node_qualname(NodeAnalysis[str|None]):
    dependencies = (node_enclosing_scope, )

class unreachable_nodes(ModuleAnalysis[set[ast.AST]]):
    ...

class _Beniget(passmanager.ModuleAnalysis):
    def do_pass(self, node:ast.Module):
        mod: passmanager.Module = self.passmanager.module
        modname = mod.modname
        if mod.is_package:
            modname += '.__init__'
        visitor = DefUseChains(mod.filename, modname, is_stub=mod.is_stub)
        visitor.visit(node)        
        return {'chains':visitor.chains, 
                'locals':visitor.locals,
                'builtins': visitor._builtins}

class def_use_chains(passmanager.ModuleAnalysis):
    dependencies = (_Beniget, )
    def do_pass(self, node: ast.Module) -> dict[ast.AST, beniget.Def]:
        return self.beniget_analsis['chains']

class use_def_chains(passmanager.ModuleAnalysis):
    dependencies = (def_use_chains, )
    def do_pass(self, node: ast.Module) -> dict[ast.AST, list[beniget.Def]]:
        return UseDefChains(self.def_use_chains).chains

class method_resoltion_order(passmanager.ClassAnalysis):
    dependencies = (passmanager.modules, )
    
    @classmethod
    def prepare(self, node):
        raise NotImplementedError()

class locals_map(passmanager.NodeAnalysis):
    dependencies = (_Beniget, )
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls):
        # There is two version of this analysis, when include_inherited=True
        # a dependency is added...
        if cls.include_inherited:
            cls.dependencies += (method_resoltion_order, ) 

        super().prepareClass()

    def do_pass(self, node: ast.AST) -> dict[str, list[beniget.Def]]:
        locals_list = self.beniget_analsis['locals'][node]
        locals_dict: dict[str, list[beniget.Def]] = {}
        for d in locals_list:
            locals_dict.setdefault(d.name(), []).append(d)
        return locals_dict

class ivars_map(passmanager.ClassAnalysis):
    dependencies = (def_use_chains, )
    optionalParameters = {'include_inherited': False}
    
    @classmethod
    def prepareClass(cls):
        # There is two version of this analysis, when include_inherited=True
        # a dependency is added...
        if cls.include_inherited:
            raise NotImplementedError() # TODO

        return super().prepareClass()
    
    def do_pass(self, node: ast.ClassDef) -> dict[str, list[beniget.Def]]:
        return _compute_ivars(self.def_use_chains, node)

class get_submodule(ModuleAnalysis[Module | None]):
    dependencies = (passmanager.modules, )
    requiredParameters = ('name',)

    def do_pass(self, node: ast.Module) -> Module | None:
        modname = self.passmanager.module.modname
        submodule_name = f'{modname}.{self.name}'
        return self.modules.get(submodule_name)

class get_ivar(ClassAnalysis[list[ast.AST]]):
    requiredParameters = ('name',)
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls): # dynamic dependenciess
        cls.dependencies = (ivars_map(include_inherited=cls.include_inherited), )
        super().prepareClass()

    def do_pass(self, node: ast.ClassDef) -> list[ast.AST]:
        return self.ivars_map[self.name]

class get_local(NodeAnalysis[list[ast.AST]]):
    requiredParameters = ('name',)
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls): # dynamic dependencies
        cls.dependencies += (locals_map(include_inherited=cls.include_inherited), )
        super().prepareClass()
    
    def prepare(self, node: ast.AST):
        super().prepare(node)
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            raise TypeError(f'expected module or class, got {node}')

    def do_pass(self, node: ast.AST) -> list[ast.AST]:
        ...

class get_attribute(NodeAnalysis[list[ast.AST]]):
    """
    >>> pm.gather(get_attribute(name='var', include_ivars=True), node)
    """
    requiredParameters = ('name',)
    optionalParameters = {'ignore_locals':False,
                          'filter_unreachable':True,
                          'include_ivars':False,
                          'include_inherited':True}

    @classmethod
    def prepareClass(cls):
        cls.dependencies += (get_submodule, )
        if not cls.ignore_locals:
            cls.dependencies += (get_local(include_inherited=cls.include_inherited), )
        if cls.include_ivars:
            cls.dependencies += (get_ivar(include_inherited=cls.include_inherited), )
        return super().prepareClass()

    def prepare(self, node: ast.AST):
        super().prepare(node)
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            raise TypeError(f'expected module or class, got {node}')

    def do_pass(self, node: ast.AST) -> list[ast.AST]:
        values: list[beniget.Def] = []
        if not self.ignore_locals:
            if self.include_ivars and isinstance(node, ast.ClassDef):
                values = self.passmanager.gather(get_ivar(name=self.name, include_inherited=self.include_inherited), node)
            if not values:
                values = self.passmanager.gather(get_local(name=self.name, include_inherited=self.include_inherited), node)
            values = _softfilter_defs(values, # type:ignore
                                        unreachable=self.filter_unreachable, 
                                        killed=True)
        else:
            values = []
        if not values and isinstance(node, ast.Module) and self.passmanager.module.is_package:
            # a sub-package
            sub = self.passmanager.gather(get_submodule(name=self.name))
            if sub is not None:
                return [sub.node]
        if values:
            return values

        raise exceptions.StaticAttributeError(node, attr=self.name, 
                                   filename=self.passmanager.module.filename)

class parsed_imports(ModuleAnalysis[Mapping[ast.alias, ImportInfo]]):
    """
    Maps each ast.alias in the module to their ImportInfo counterpart.
    """
    def do_pass(self, node: ast.Module) -> Mapping[ast.alias, ImportInfo]:
        return ParseImportedNames(self.passmanager.module.modname, 
                                  is_package=self.passmanager.module.is_package).visit_Module(node)

class definitions_of_imports(ModuleAnalysis[Mapping[ast.alias, list[ast.AST]]]):
    dependencies = (parsed_imports, passmanager.modules)

    def do_pass(self, node: ast.Module) -> Mapping[ast.alias, list[ast.AST]]:

        self.modules


# simple test stuff

class simple_symbol_table(ModuleAnalysis[dict[str, list[ast.AST]]], ast.NodeVisitor):
    "A simbol table, for the module level only"
    def do_pass(self, node: ast.Module) -> dict[str, list[ast.AST]]:
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
    dependencies = (simple_symbol_table, ancestors)

    def do_pass(self, node: ast.Name) -> ast.AST | None:
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
    
    def do_pass(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

# An analysis that has a transform in it's dependencies

class has_optional_parameters(NodeAnalysis):
    optionalParameters = {'filter': False, 'inherited': False}
    def do_pass(self, node): pass

class has_required_parameters(NodeAnalysis):
    requiredParameters = ('name', 'thing', )
    def do_pass(self, node): pass

class has_both_required_and_optional_parameters(NodeAnalysis):
    requiredParameters = ('name', 'thing', )
    optionalParameters = {'filter': False, 'inherited': False}
    def do_pass(self, node): pass

class requires_modules(NodeAnalysis):
    dependencies = (passmanager.modules, )
    def do_pass(self, node): pass

class has_dynamic_dependencies(NodeAnalysis):
    dependencies = (has_required_parameters, )
    optionalParameters = {'project_wide': False}
    @classmethod
    def prepareClass(cls):
        if cls.project_wide:
            cls.dependencies += (requires_modules, )
    def do_pass(self, node): pass

class literal_ones_count(ModuleAnalysis[int], ast.NodeVisitor):
    "counts the number of literal '1'"
    dependencies = (transform_trues_into_ones, 
                    # this analysis is useless but it should not 
                    # crash because it's listed after the transform.
                    # TODO: Actually since the transform preserves the class_count analysis,
                    # both orders shouls be supported. But that might be hard to implement at this time.
                    class_count) 
    
    def visit_Constant(self, node):
        if node.value == 1:
            self.result += 1

    def do_pass(self, node: ast.AST) -> int:
        self.result = 0
        self.visit(node)
        return self.result

# an invalid analysis because it has an analysis BEFORE a transformation

class still_valid_analysis_dependencies_order(ModuleAnalysis[int]):
    "this analysis is not valid actually, not everything is ok."
    dependencies = (node_list, transform_trues_into_ones)

    def do_pass(self, node: ast.AST) -> int:
        return 1

# an invalid analysis because it has not do_pass() method

class invalid_analysis_no_do_pass_method(ModuleAnalysis[int]):
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
            def do_pass(self, node: ast.AST) -> list[str]:
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
    
    def test_analysis_valid_dependencies_order(self):
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        
        pm.gather(still_valid_analysis_dependencies_order, pm.modules['test'].node)
    
    def test_analysis_invalid_no_do_pass_method(self):
        src = 'pass'
        pm = PassManager()
        pm.add_module(Module(
            ast.parse(src), 'test', 'test.py', code=src, 
        ))
        with self.assertRaises(Exception):
            pm.gather(invalid_analysis_no_do_pass_method, pm.modules['test'].node)

    def test_analysis_with_analysis_and_transforms_dependencies(self):
        ...

    def test_transformation_with_analysis_dependencies(self):
        # the test transformation would be to remove unused imports for instance.
        # this will require the def-use chains.
        ...
    
    def test_transformation_with_transforms_dependencies(self):
        # so for instance
        ...
    
    def test_transformation_with_analysis_and_transforms_dependencies(self):
        ...
    
    def test_transformation_with_unsuported_cyclic_dependencies(self):
        ...
    
    def test_transformation_with_suported_cyclic_dependencies(self):
        # this uses 
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
    
    def test_node_to_module_analysis_promotion(self):
        ...
    
    def test_function_to_project_analysis_promotion(self):
        ...
    
    def test_class_to_project_analysis_promotion(self):
        ...
    
    def test_module_to_project_analysis_promotion(self):
        ...
    
    def test_node_to_project_analysis_promotion(self):
        ...
    

