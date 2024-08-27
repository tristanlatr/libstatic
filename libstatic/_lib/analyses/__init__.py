"""
Collection of analyses. 
"""
from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

import sys
import ast

from libstatic._lib.imports import ParseImportedNames, ImportInfo
from libstatic._lib.ivars import _compute_ivars
from libstatic._lib import exceptions
from libstatic._lib.passmanager import (Module, NodeAnalysis, ClassAnalysis, ModuleAnalysis)
from libstatic._lib import passmanager

from beniget.standard import DefUseChains, UseDefChains # type: ignore
import beniget # type: ignore


ancestors = passmanager.ancestors
modules = passmanager.modules

class expand_expr(NodeAnalysis['str | None']):
    ...

class node_ancestor(NodeAnalysis[ast.AST]):
    """
    First node ancestor of class C{klass}. 

    >>> from pprint import pprint
    >>> mod = ast.parse('v = lambda x: x+1; w = 2')
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(mod, 'test'))
    >>> const = mod.body[-1].value
    >>> pm.gather(node_ancestor(klass=ast.Assign), const).__class__.__name__
    'Assign'

    @param klass: type or tuple of types.
    """
    dependencies = (ancestors, )
    requiredParameters = ('klass', )

    def doPass(self, node: ast.AST) -> ast.AST:
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

class node_enclosing_scope(NodeAnalysis['ast.AST | None']):
    """
    Get the first enclosing scope of this use or definition.
    Returns None only of the definition is a Module.

    >>> mod = ast.parse('v = lambda x: x+1; w = 2')
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(mod, 'test'))
    >>> lb = pm.gather(node_enclosing_scope, mod.body[0].value.body)
    >>> lb.__class__.__name__
    'Lambda'
    >>> pm.gather(node_enclosing_scope, lb).__class__.__name__
    'Module'
    """
    dependencies = (node_ancestor.bind(klass=(
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
    
    def doPass(self, node: ast.AST) -> ast.AST | None:
        if isinstance(node, ast.Module):
            return None
        return self.node_ancestor

if sys.version_info >= (3, 10):
    from libstatic._lib.scopetree import Builder as ScopeTreeBuilder, Scope
    class scope_tree(ModuleAnalysis[dict[ast.AST, Scope]]):
        """
        Collect scope information as well as which scope uses which name. 

        >>> src = '''
        ... class C:
        ...     def foo(self, a = blah):
        ...         global x
        ...         x = a
        ... '''
        >>> mod = ast.parse(src)
        >>> pm = passmanager.PassManager()
        >>> pm.add_module(passmanager.Module(mod, 'test'))
        >>> scopes = pm.gather(scope_tree, mod)
        >>> from .. import scopetree
        >>> print(scopetree.dump(scopes.values()))
        GlobalScope('<globals>'): L=['C']; U={}
        ClassScope('C'): L=['foo']; U={'blah': None}
            FunctionScope('foo'): L=['a', 'self']; G=['x']; U={'a': FunctionScope('foo')}
        <BLANKLINE>

        """
        def doPass(self, node: ast.Module) -> dict[ast.AST, Scope]:
            builder = ScopeTreeBuilder()
            builder.build(node)
            return builder.scopes

class node_qualname(NodeAnalysis['str | None']):
    """
    Get the qualified name of the definition. If the node is not
    the definition of a name it will raise an error.
    """
    dependencies = (node_enclosing_scope, )

class literal_eval(NodeAnalysis[object]):
    """
    @param known_values: Some values we know already, typically C{sys.version_info}. 
        Type is L{dict} items as tuple (since parameters needs to be hashable).
    @type known_values: tuple[tuple[str, object], ...]
    """
    optionalParameters = {
        'known_values' : None,
        'follow_imports' : False,
    }

class unreachable_nodes(ModuleAnalysis[set[ast.AST]]):
    ...

class _Beniget(passmanager.ModuleAnalysis[DefUseChains]):
    """
    A wrapper for U{https://github.com/pyforks/beniget-ng}

    Until U{https://github.com/serge-sans-paille/beniget/pull/93} is merges we can't use upstream version.
    """
    def doPass(self, node:ast.Module) -> DefUseChains:
        mod: passmanager.Module = self.ctx.module
        modname = mod.modname
        if mod.is_package:
            modname += '.__init__'
        visitor = DefUseChains(mod.filename, modname, is_stub=mod.is_stub)
        visitor.visit(node)        
        return visitor

class def_use_chains(passmanager.ModuleAnalysis):
    """
    Results in a mapping from nodes to their L{beniget.Def} instances, 
    which maps all uses of definitions.
    """
    dependencies = (_Beniget, )
    def doPass(self, _: ast.Module) -> dict[ast.AST, beniget.Def]:
        return self._Beniget.chains

class use_def_chains(passmanager.ModuleAnalysis):
    """
    Results in a mapping from use nodes to the list of potential definitions
    as list of L{beniget.Def} instances.
    """
    dependencies = (def_use_chains, )
    def doPass(self, node: ast.Module) -> dict[ast.AST, list[beniget.Def]]:
        return UseDefChains(self.def_use_chains).chains

class method_resoltion_order(passmanager.ClassAnalysis):
    dependencies = (passmanager.modules, )
    
    @classmethod
    def prepare(self, node):
        raise NotImplementedError('not there yet')

class locals_map(passmanager.NodeAnalysis[dict[str, list[beniget.Def]]]):
    """
    >>> src = '''
    ... v = 9
    ... def f(a, b) -> object:...
    ... class C(object):...
    ... '''
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(ast.parse(src), 'test'))
    >>> pm.gather(locals_map, pm.modules['test'].node)
    {'v': [<ast.Name object at ...> -> ()], 'f': [<ast.FunctionDef object at ...> -> ()], 'C': [<ast.ClassDef object at ...> -> ()]}
    """
    dependencies = (_Beniget, )
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls): # dynamic dependencies
        if cls.include_inherited:
            cls.dependencies += (method_resoltion_order, ) 

        super().prepareClass()

    def doPass(self, node: ast.AST) -> dict[str, list[beniget.Def]]:
        locals_list = self._Beniget.locals[node]
        locals_dict: dict[str, list[beniget.Def]] = {}
        for d in locals_list:
            locals_dict.setdefault(d.name(), []).append(d)
        return locals_dict

class ivars_map(passmanager.ClassAnalysis):
    """
    >>> src = '''
    ... class c:
    ...     def __init__(self, x, a):
    ...         self.x = a
    ...         self._a = x
    ... '''
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(ast.parse(src), 'test'))
    >>> pm.gather(ivars_map, pm.modules['test'].node.body[0])
    {'x': [<ast.Attribute ...> -> ()], '_a': [<ast.Attribute ...> -> ()]}
    """
    dependencies = (def_use_chains, )
    optionalParameters = {'include_inherited': False}
    
    @classmethod
    def prepareClass(cls): # dynamic dependencies
        if cls.include_inherited:
            cls.dependencies += (method_resoltion_order, )

        return super().prepareClass()
    
    def doPass(self, node: ast.ClassDef) -> dict[str, list[beniget.Def]]:
        return _compute_ivars(self.def_use_chains, node)

class get_submodule(ModuleAnalysis['Module | None']):
    """
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(ast.parse('pass'), 'test', is_package=True))
    >>> pm.add_module(passmanager.Module(ast.parse('pass'), 'test.framework', is_package=True))
    >>> pm.add_module(passmanager.Module(ast.parse('pass'), 'test.framework._impl'))
    >>> pm.add_module(passmanager.Module(ast.parse('pass'), 'test.framework.stuff', is_package=True))
    >>> pm.gather(get_submodule(name='stuff'), pm.modules['test.framework'].node)
    Module(node=<ast.Module object at ...>, modname='test.framework.stuff', filename=None, is_package=True, is_stub=False, code=None)
    """
    dependencies = (passmanager.modules, )
    requiredParameters = ('name',)

    def doPass(self, node: ast.Module) -> Module | None:
        modname = self.ctx.module.modname
        submodule_name = f'{modname}.{self.name}'
        return self.modules.get(submodule_name)

class get_ivar(ClassAnalysis[list[ast.AST]]):
    """
    >>> src = '''
    ... class c:
    ...     def __init__(self, x, a):
    ...         self.x = a
    ...         self._a = x
    ... '''
    >>> pm = passmanager.PassManager()
    >>> pm.add_module(passmanager.Module(ast.parse(src), 'test'))
    >>> pm.gather(get_ivar(name='x'), pm.modules['test'].node.body[0])
    [<ast.Attribute ...> -> ()]
    """
    requiredParameters = ('name',)
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls): # dynamic dependenciess
        cls.dependencies = (ivars_map.bind(include_inherited=cls.include_inherited), )
        super().prepareClass()

    def doPass(self, node: ast.ClassDef) -> list[beniget.Def]:
        return self.ivars_map[self.name]

class get_local(NodeAnalysis[list[ast.AST]]):
    requiredParameters = ('name',)
    optionalParameters = {'include_inherited': False}

    @classmethod
    def prepareClass(cls): # dynamic dependencies
        cls.dependencies += (locals_map.bind(include_inherited=cls.include_inherited), )
        super().prepareClass()
    
    def prepare(self, node: ast.AST):
        super().prepare(node)
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            raise TypeError(f'expected module or class, got {node}')

    def doPass(self, node: ast.AST) -> list[ast.AST]:
        return [] # TODO

class get_attribute(NodeAnalysis[list[ast.AST]]):
    """
    >>> # pm.gather(get_attribute(name='var', include_ivars=True), node)
    """
    requiredParameters = ('name',)
    optionalParameters = {'ignore_locals':False,
                          'filter_unreachable':True,
                          'include_ivars':False,
                          'include_inherited':True}

    @classmethod
    def prepareClass(cls):
        cls.dependencies += (get_submodule, )
        if cls.filter_unreachable:
            cls.dependencies += (unreachable_nodes, )
        if not cls.ignore_locals:
            cls.dependencies += (get_local.bind(include_inherited=cls.include_inherited), )
        if cls.include_ivars:
            cls.dependencies += (get_ivar.bind(include_inherited=cls.include_inherited), )
        return super().prepareClass()

    def prepare(self, node: ast.AST):
        super().prepare(node)
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            raise TypeError(f'expected module or class, got {node}')

    def doPass(self, node: ast.AST) -> list[ast.AST]:
        values: list[beniget.Def] = []
        if not self.ignore_locals:
            if self.include_ivars and isinstance(node, ast.ClassDef):
                values = self.passmanager.gather(get_ivar.bind(name=self.name, include_inherited=self.include_inherited), node)
            if not values:
                values = self.passmanager.gather(get_local.bind(name=self.name, include_inherited=self.include_inherited), node)
            
            # "soft" filter killed and unreachable definitions
            values = list(filter(lambda v: v.islive, ov:=values)) or ov
            if self.filter_unreachable:
                values = list(filter(lambda v: v.node not in self.unreachable_nodes, ov:=values)) or ov

        else:
            values = []
        
        if not values and isinstance(node, ast.Module) and self.ctx.module.is_package:
            # a sub-package
            sub = self.passmanager.gather(get_submodule.bind(name=self.name), node)
            if sub is not None:
                return [sub.node]
        if values:
            return values

        raise exceptions.StaticAttributeError(node, attr=self.name, 
                                   filename=self.ctx.module.filename)

class parsed_imports(ModuleAnalysis[Mapping[ast.alias, ImportInfo]]):
    """
    Maps each ast.alias in the module to their ImportInfo counterpart.
    """
    def doPass(self, node: ast.Module) -> Mapping[ast.alias, ImportInfo]:
        return ParseImportedNames(self.ctx.module.modname, 
                                  is_package=self.ctx.module.is_package).visit_Module(node)

class definitions_of_imports(ModuleAnalysis[Mapping[ast.alias, list[ast.AST]]]):
    dependencies = (parsed_imports, passmanager.modules)

    def doPass(self, node: ast.Module) -> Mapping[ast.alias, list[ast.AST]]:
        assert self.modules
        return {}
