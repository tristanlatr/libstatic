"""
This module provides a framework for L{ast} pass management. 
It's a building block to write static analyzer or compiler for Python. 

There are two kinds of passes: transformations and analysis.
    * L{ModuleAnalysis} and L{NodeAnalysis} are to be
      subclassed by any pass that collects information about the AST.
    * L{PassManager.gather} is used to gather (!) the result of an analyses on an AST node.
    * L{Transformation} is to be sub-classed by any pass that updates the AST.
    * L{PassManager.apply} is used to apply (sic) a transformation on an AST node.

To write an analysis: 

    - Subclass one of the analysis class cited above.
    - List passes required by yours in the L{Analysis.dependencies} tuple, 
      they will be built automatically and stored in the attribute with the corresponding name.
    - Write your analysis logic inside the L{Analysis.doPass} method. The analysis result must be returned by
      the L{Analysis.doPass} method or an exeception raised.
    - Use it either from another pass’s C{dependencies}, or through the L{PassManager.gather} function.

B{Example}: In the following code snippets we'll look at how to write a type inference function that
understand literal values as well as builtins calls like L{list()} with a relatively high level of confidence
we're not misinterpreting a symbol name.   

Start by coding the logic of your lower-level analysis, which in our case computes the locals for a given scope:

>>> import ast
>>> class CollectLocals(ast.NodeVisitor):
...    "Compute the set of identifiers local to a given node."
...    def __init__(self):
...        self.Locals = set()
...        self.NonLocals = set()
...    def visit_FunctionDef(self, node):
...        # no recursion
...        self.Locals.add(node.name)
...    visit_AsyncFunctionDef = visit_FunctionDef
...    visit_ClassDef = visit_FunctionDef
...    def visit_Nonlocal(self, node):
...        self.NonLocals.update(name for name in node.names)
...    visit_Global = visit_Nonlocal
...    def visit_Name(self, node):
...        if isinstance(node.ctx, ast.Store) and node.id not in self.NonLocals:
...            self.Locals.add(node.id)
...    def visit_arg(self, node):
...        self.Locals.add(node.arg)
...    def skip(self, _):
...        pass
...    visit_SetComp = visit_DictComp = visit_ListComp = skip
...    visit_GeneratorExp = skip
...    visit_Lambda = skip
...    def visit_Import(self, node):
...        for alias in node.names:
...            base = alias.name.split(".", 1)[0]
...            self.Locals.add(alias.asname or base)
...    def visit_ImportFrom(self, node):
...        for alias in node.names:
...            self.Locals.add(alias.asname or alias.name)

Then wraps it inside an Analysis subclass:

>>> class node_locals(NodeAnalysis[set[str]]):
...     def doPass(self, node):
...         visitor = CollectLocals()
...         visitor.generic_visit(node)
...         return visitor.Locals

Now let's try out our analysis, here we are hard coding the source code for testing purposes
but in real life you can use the L{Finder}. 

>>> src = '''
... import sys, os, set
... from twisted.python import reflect
... import twisted.python.filepath
... from twisted.python.components import registerAdapter
... foo, foo2 = 123, int()
... def doFoo():
...   import datetime
... class Foo:
...   x = 0
...   list = lambda x: True
...   y = list() or set()
...   def __init__(self):
...     self.a = [a for a in list()]
... _foo = "boring internal details"
... '''

This is where is gets interesting, how to actually use the framework

>>> pm = PassManager() # Create a PassManager instance
>>> pm.add_module(Module(ast.parse(src), 'testmodulename')) # Add your module(s) to it

When you add a module it's recorded in the L{PassManager.modules} mapping. It's a special mapping
that allows accessing values by module name or by any node contained in the module tree.

>>> testnode = pm.modules['testmodulename'].node
>>> testclass = next(n for n in testnode.body if isinstance(n, ast.ClassDef) and n.name == 'Foo')
>>> testmethod = next(n for n in testclass.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')

Gather analysis result with L{PassManager.gather}.

>>> print(sorted(pm.gather(node_locals, testnode) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted']

>>> print(sorted(pm.gather(node_locals, testclass) ))
['__init__', 'list', 'x', 'y']

>>> print(sorted(pm.gather(node_locals, testmethod) ))
['self']

Now let's not forget about our goal: builtins calls inference. So in order to be relatively sure we're not 
misinterpreting builtins names, we nee to know all accessible names from any expression nodes. If the name 'list' is not
declared it means it's an actual builtins call. 

Let's use one of the provided analysis: L{analyses.node_enclosing_scope} as another dependency.

>>> from .. import analyses

Note how the analyses can both be used as instance attribute, in which case they are run on the current node; 
or as C{self.passmanager.gather()} argument to run them on any other nodes.

>>> ast_scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, 
...     ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp, ast.Lambda, ast.Module)
>>> class declared_names(NodeAnalysis[set[str]]):
...     "Collects all the declared names accessible from the current node"
...     dependencies = (node_locals, analyses.node_enclosing_scope, )
...     def doPass(self, node):
...         declared = set(); i = 0; 
...         scope = self.node_enclosing_scope if not isinstance(node, ast_scopes) else node
...         while scope is not None:                      # first iteration. 
...             if not isinstance(scope, ast.ClassDef) or i==0:
...                 declared.update(self.passmanager.gather(node_locals, scope))
...             scope = self.passmanager.gather(analyses.node_enclosing_scope, scope)
...             i += 1
...         return declared

Let's try out this new analysis.

>>> print(sorted(pm.gather(declared_names, testnode) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted']

>>> print(sorted(pm.gather(declared_names, testclass) ))
['Foo', '__init__', '_foo', 'doFoo', 'foo', 'foo2', 'list', 'os', 'reflect', 'registerAdapter', 'set', 'sys', 'twisted', 'x', 'y']

>>> print(sorted(pm.gather(declared_names, testmethod) ))
['Foo', '_foo', 'doFoo', 'foo', 'foo2', 'os', 'reflect', 'registerAdapter', 'self', 'set', 'sys', 'twisted']

Looks good... Now putting everything together in the C{infer_type} analysis:

>>> supported_builtins_calls = { # we limit ourselves to a subset of builtins calls.
...     'list': list, 
...     'dict': dict, 
...     'set': set, 
...     'tuple': tuple, 
...     'str': str, 
...     'frozenset': frozenset, 
...     'int': int
... } # that's not exhaustive but it's a start

>>> class infer_type(NodeAnalysis[type]):
...     '''
...     This class tries to infer the types of literal expression as 
...     well builtins calls like dict(). 
...     '''
...     dependencies = (declared_names, )
...     def doPass(self, node) -> type | None:
...         if (isinstance(node, ast.Call) and 
...             isinstance(node.func, ast.Name) and 
...             node.func.id in supported_builtins_calls):
...             if node.func.id not in self.declared_names:
...                 return supported_builtins_calls[node.func.id]
...             return None # the builtin name is actually shadowed so we don't know...
...         try:
...             # try the node as a literal expression
...             return type(ast.literal_eval(node))
...         except Exception:
...             return None

>>> testintcall = next(n for n in ast.walk(testnode) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'int')
>>> testsetcall = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'set')
>>> testlistcall1 = next(n for n in ast.walk(testclass) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testlistcall2 = next(n for n in ast.walk(testmethod) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'list')
>>> testliteralstring = next(n for n in ast.walk(testnode) if isinstance(n, ast.Constant) and isinstance(n.value, str))

We now have a manner to statically infer types of builtins calls with relatively high confidence 
(It could be improved, this is just an example after all).

>>> pm.gather(infer_type, testintcall)
<class 'int'>
>>> pm.gather(infer_type, testsetcall) is None # fails because 'set' is an imported name
True
>>> pm.gather(infer_type, testlistcall1) is None # fails because 'list' is defined in class body
True
>>> pm.gather(infer_type, testlistcall2)
<class 'list'>
>>> pm.gather(infer_type, testliteralstring)
<class 'str'>

Sometime it is practical for an analysis to have different behaviour depending on optional or required parameters.
Instead of writing several L{Analysis} subclass calling un underlying function with different parameters, write an analysis taking parameters: 
    
    - List all required parameters names in the C{requiredParameters} tuple.
    - List all optional parameters in the C{optionalParameters} dict; 
      keys are parameters names are values are the default values in case none is provided.
    - To create your parameterized analysis simply call your analysis class with keywords coresponding to your
      parameters. This will create a devired subclass with the parameters set at class level.
    
To write a transformation...

    - Subclass L{Transformation} class. Keep in mind that a transformation is always run at the module level.
    - List passes required by yours in the L{Analysis.dependencies} tuple, 
      they will be built automatically and stored in the attribute with the corresponding name.
    - Write your analysis logic inside the L{Analysis.doPass} method. The analysis result must be returned by
      the L{Analysis.doPass} method or an exeception raised.
    - Use it either from another pass’s C{dependencies}, or through the L{PassManager.gather} function.


    
"""
from __future__ import annotations

from ._passmanager import PassManager, Pass, Analysis, ModuleAnalysis, NodeAnalysis, Transformation
from ._passmanager import ClassAnalysis, FunctionAnalysis
from ._modules import Module, ModuleCollection
from ._passmanager import ancestors, modules, GetProxy, PassContext

__all__ = ('PassManager', 'Pass', 'Analysis', 'ModuleAnalysis', 'NodeAnalysis', 'Transformation',
           'ClassAnalysis', 'FunctionAnalysis',
           'Module', 'ModuleCollection', 
           'ancestors', 'modules', 'GetProxy', 'PassContext')