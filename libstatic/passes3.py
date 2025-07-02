# TODO: All of the implementations should go into submodules and then exported

from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

import ast

import beniget

from . import passmanager3 as passmanager

# from libstatic._lib.imports import ParseImportedNames, ImportInfo
# from libstatic._lib.ivars import _compute_ivars
# from libstatic._lib import exceptions

################## Transformations

from libstatic._lib.transform import Transform
@passmanager.transformation(on=passmanager.Tree)
def normalize__all__(_, node: passmanager.Tree):
    transformer = Transform()
    transformer.transform(node.root)
    yield 'update', transformer.update

################## Node Ancestors related analyses

@passmanager.analysis(on=passmanager.Tree)
def ancestors(_, node: passmanager.Tree):
    """
    See L{beniget.Ancestors}.

    >>> pm = passmanager.PassManager()
    >>> pm.add(passmanager.MTree(ast.parse('v = lambda x: x+1'), 'test'))
    >>> r = pm.gather(ancestors, 'test')
    >>> isinstance(r, beniget.Ancestors)
    True
    """
    ancestors = beniget.Ancestors()
    ancestors.visit(node.root)
    yield 'result', ancestors

def fix_ancestors(ans: beniget.Ancestors, source: ast.AST, target: ast.AST):
    """
    Register the C{target} node in the given L{beniget.Ancestors} instance with
    the same parents as the C{source} node, then traverse the target tree to register
    all child nodes. This is useful in order to update in place the ancestors while 
    transforming the tree to avoid recomputation of the ancestors 
    again and again for each transforms.

    See: L{ancestors}. 
    
    @param ans: The L{beniget.Ancestors} instance.
    @param source: An existing node in the ancestors 
        that should have the same parents as the fixed-up node.
    @param target: A node to fix-up.
    """
    old_current, ans._current = ans._current, ans.parents(source)
    try:
        ans.visit(target)
    finally:
        ans._current = old_current

def prune_ancestors(ans: beniget.Ancestors, node: ast.AST):
    """
    Remove the given node and all it's child nodes from the ancestors instance.
    """
    for n in ast.walk(node):
        del ans._parents[n]

@passmanager.analysis(on=ast.AST, dependencies=[ancestors])
def node_ancestor(c: passmanager.Connector, node: ast.AST, *, 
                  klass: type | tuple[type, ...]):
    """
    First node ancestor of class C{klass}. 

    >>> pm = passmanager.PassManager()
    >>> pm.add(passmanager.MTree(ast.parse('v = lambda x: x+1'), 'test'))
    >>> lbody = pm.trees['test'].root.body[0].value.body
    >>> r = pm.gather(node_ancestor(ast.Assign), 'test', lbody)
    >>> r.__class__.__name__
    'Assign'

    @param klass: type or tuple of types.
    """
    ans: beniget.Ancestors = c.deps.ancestors
    
    if isinstance(klass, type) and issubclass(klass, ast.Module):
        # special case module access for speed.
        # don't forget klass can be a tuple
        parents = ans.parents(node)
        try:
            mod = next(iter(parents))
        except StopIteration:
            pass
        else:
            if isinstance(mod, klass):
                yield 'result', mod  # type: ignore
                return
    # otherwise defers to Ancestors.parentInstance()
    yield 'result', ans.parentInstance(node, klass)

@passmanager.analysis(on=ast.AST, 
    dependencies=[node_ancestor((
            ast.SetComp,
            ast.DictComp,
            ast.ListComp,
            ast.GeneratorExp,
            ast.Lambda,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Module,
        ))])
def node_enclosing_scope(c: passmanager.Connector, node: ast.AST):
    """
    Get the first enclosing scope of this use or definition.
    Returns None only of the definition is a Module.

    >>> mod = ast.parse('v = lambda x: x+1')
    >>> pm = passmanager.PassManager()
    >>> pm.add(passmanager.MTree(mod, 'test'))
    >>> lb = pm.gather(node_enclosing_scope, 'test', mod.body[0].value.body)
    >>> lb.__class__.__name__
    'Lambda'
    >>> pm.gather(node_enclosing_scope, 'test', lb).__class__.__name__
    'Module'
    """

    if isinstance(node, ast.Module):
        yield 'result', None
        return
    yield 'result', c.deps.node_ancestor

################## Scope tree analysis


################## Qualnames related analysis


################## Variables and attribute access related analysis


################## Def-Use/Use-Def chains related analysis

@passmanager.analysis(on=passmanager.Tree)
def def_use_chains(_, node: passmanager.Tree):
    """
    See L{beniget.DefUseChains}.

    >>> pm = passmanager.PassManager()
    >>> pm.add(passmanager.MTree(ast.parse('v = 1; v += 2'), 'test'))
    >>> r = pm.gather(def_use_chains, 'test')
    >>> isinstance(r, beniget.DefUseChains)
    True
    """
    visitor = beniget.DefUseChains(node.identifier)
    visitor.visit(node.root)
    yield 'result', visitor

@passmanager.analysis(on=passmanager.Tree, dependencies=[def_use_chains])
def use_def_chains(c: passmanager.Connector, _):
    """
    See L{beniget.UseDefChains}.

    >>> pm = passmanager.PassManager()
    >>> pm.add(passmanager.MTree(ast.parse('v = 1; v += 2'), 'test'))
    >>> r = pm.gather(use_def_chains, 'test')
    >>> isinstance(r, beniget.UseDefChains)
    True
    """
    yield 'result', beniget.UseDefChains(c.deps.def_use_chains)

################## Imports related analysis



################## Running doctests
# this is unfortunate but doctest doesn't find the 
# code under test except if it's defined under another name.
# TODO: I did not expected this and I should really find a solution...

class _DocTests:
    """
    Create the nessary placeholder objects for doctest to discover 
    the test cases accros the decorated functions.
    """
    def _doctest(passe):
        fn = lambda: None
        fn.__doc__ = passe.__doc__
        return fn

    import sys, inspect
    for name, member in inspect.getmembers(sys.modules[__module__]):
        if isinstance(member, passmanager.PassPrototype):
            vars()[name] = _doctest(member)
