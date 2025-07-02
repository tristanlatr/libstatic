# The libstatic framework 

-- for syntax tree analysis and manipulation --

Summary
-------

This package introduces a framework designed to help developers work with the abstract syntax tree (AST) analysis and rewriting.
It is an implementation of the pass manager architectural pattern in compiler design, 
comparable to the ones found in [LLVM](https://llvm.org/docs/WritingAnLLVMPass.html#what-passmanager-does), 
[GCC](https://gcc.gnu.org/onlinedocs/gccint/Pass-manager.html) or [pythran](https://pythran.readthedocs.io/en/latest/TUTORIAL.html). 

The main focus beeing to design a modular, well tested, well documented framework to integrate existing 
static analysis code into a reusable fashion based on software development best practices. 

In contrast from existing passmanagers, this one works on an entire forest of Python modules and packages. 

Motivation
----------

Why should this work be done? 

The Python ecosystem needs a reusable framework to work with the abstract syntaxt tree that
provides high-level functions to resolve common static analyzer needs. Such as expanding
wildcard imports, resolving the MRO, explicitely add auto generated code by dataclasses or attrs, etc... 

The pass manager architecture allows for a simple and yet flexible interface for analyzing and rewriting code. 


>>>>>> 

#### Wildcard imports

Issues like wildcard imports and dynamically built ``__all__`` variables are a major pain 
for many static analysis tools: every tool needs to implement their own 
resolving logic which is not always inline with others.

Both pyright and mypy support dynamically built ``__all__`` variables as well as wildcard imports, this makes the python feature 
used across a lot of code, including stub modules. So the attempt to provide type inference based on typeshed stubs is irrelevant 
for those without the ability to statically resolve wildcards imports (There are currently more than 200 wildcard import in the typeshed stubs).


#### Dataclasses and attrs

Dataclasses and friends are a very efficient manner to declare classes; 
but they add methods that were not explicitely written by the developpers.
In static analysis, this often means that the tool will require a custom 
plugin or hook in order to understand it right. 

The framework takes a different approach, it offers a transformation that will 
make explicit all generated methods. See the [undataclass project](https://github.com/treyhunner/undataclass).


Non-Goals of the framework
--------------------------

- Whole-program refactoring utilities, for instance renaming a module/class/method etc. While it could be included, some existing libraries are doing a good
  job for these duties. You can use [rope](https://rope.readthedocs.io/en/latest/library.html#rename) or [libcst](https://libcst.readthedocs.io/) for instance. 
  The transformation interface is designed with optimization and normalization in mind, not cross-modules refactoring. Also, the AST does not carry the formatting
  so any attempt to write the transformed code back to file should be done with great care.
- Introspection of modules. One could write code that generates AST for a given module by inspection 
(like [astroid](https://github.com/pylint-dev/astroid/blob/v3.2.0/astroid/raw_building.py) does). But this is not in the scope and can be done on the client side.


<<<<<<<<

What is a passmanager ?
-----------------------

The pass manager is a component that orchestrates the execution 
of various passes on the code being processed with the objectives of 
effective code analysis and transformation in compilers, static analyzers, and related tools. 
**It is the primary interface of the library**. 

The pass manager is responsible to optimize analyses results so they don't have to be run unnecessarily; 
as well as arranges for the transformations to happen in the correct order, prerare the modular dependencies 
required by a pass, maintains the cache and sain by invalidate results when it needs. 

Alongside the passmanager, a collection of qualitative passes and utilities is 
provided to make development and adoption faster. 

See _Bundled analyses and transformations_ for more informations on bundled analyses.

Package structure
------------------

```tree
libstatic/
├─ passmanager/
├─ analyses/
├─ transformations/
├─ instrumentations/
├─ contrib/
├─ utils.py
├─ exceptions.py
├─ finder.py
├─ cli.py
```

Contents
--------

``passmanager/``: The core of the framework, declares:
- the passmanager `PassManager` and related classes. 
- the `Pass` class and factories: `@analysis()` and `@transformation()`
- the three-layered code abstraction model: `Forest`, `Tree`, node.

``transformations/``: Easy to use, reusable ast transformers: 
- various code normalization and desugaring
- expand wildcard imports
- undataclass / unattrs
- remove dead code
- constant folding
- produce a single assignment form of the IR

``analyses/``: Easy to use, reusable analyses: 
- chains of definition / uses accros the whole forest; making it really easy to implement
  whole-program forward or backward analysis. 
- some level of type inference
- mro graph
- class instance variables
- import graph/scc
- symbolic evaluation
- 

``instrumentations/``: Set of hooks to customize some aspects of the passmanager.

``finder.py``: Does the IO, it creates `Tree` instances based on provided path.

``utils.py``: Misc utilities.

``exceptions.py``: Expections used in the framework. 

```python
class StaticException(Exception):...
class StaticTreeNotFound(StaticException):...
class StaticNameError(StaticException):...
class StaticAttributeError(StaticException):...
class StaticTypeError(StaticException):...
class StaticTypeMismatch(StaticException):...
class StaticImportError(StaticException):...
class StaticValueError(StaticException):...
class StaticStateIncomplete(StaticException):...
class StaticCodeUnsupported(StaticException):...
class StaticAmbiguity(StaticException):...
class StaticEvaluationError(StaticException):...
class StaticUnknownValue(StaticException):...
```

``cli.py``: The command line tool for a syntax tree transformer and analyzer


Specification
-------------

### The code abstraction model

The framework does not encode the syntax tree in a particular IR. Actually, the core
of the pass manager is syntax tree agnostic: given the proper configuration it can
run on any strcuture. 

The Python language comes with a useful ``ast`` module; so much code
is already built on top of it. So the choive has his makes it a good candidate for further work.

This comes with some limitation regarding mutability of the tree: 
the node passed to the transformation function can never be itself replaced, only it's content. 


#### The node

A node is anything that is generated from the parser, such as instances of the following types: 

```
ast.Module, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Name, ast.Attribute, etc...
```

See: https://docs.python.org/3/library/ast.html#abstract-grammar


#### The tree

The ``Tree`` class encapsulate a single parsed file. It is a frozen datastructure. 

The client code is responsible to create ``Tree`` instances that carries the
parse tree over to the framework. ``Tree`` instances can be created 
manually or with the ``finder`` (discussed below).


```python
class Tree:
    root: ast.Module
    identifier: str
    attributes: frozendict
```

The attributes dictionary optionally includes metadata such as: 
- ``filename``: the file name. This shall never be used to do IO; 
  since it's the client code that is responsible create the syntax tree.
- ``is_package``: whether the tree represents the ``__init__.py`` of a package.
- ``is_stub``: whether the tree is a stub.
- ``lines``: the source code lines as a tuple or strings.

All the attributes values should be immutable constants such as: 
``tuple``, ``frozenset``, ``str``, ``bool``. 
But NOT ``dict``, ``list`` or ``set`` for instance.

#### The forest

The forest represents the whole domain of study, in Soot that would be the Scene. 


This class is not designed to be mutated by client code, nevertheless
it can be used by like a read-only collection of ``Tree``s.

```python
class Forest(Collection[Tree]):
    def get():...
    def __getitem__():...
    def __iter__():...
    def __len__():...
    def __contains__():...
```

### The PassManager API

Gather the result of an analysis on the entire forest, a tree or an AST node.

* ``PassManager.gather(analysis) -> object`` (forest-wide)
* ``PassManager.gather(analysis, tree) -> object`` (tree-wide)
* ``PassManager.gather(analysis, tree, node) -> object`` (node-wide)

Apply a transformation on a tree or an AST node. 
Returns whether the pass updated the content.

* ``PassManager.apply(transformation, tree) -> bool`` (tree-wide)
* ``PassManager.apply(transformation, tree, node) -> bool`` (node-wide)

Run an analysis or a transformation. Returns an instance of ``CompletedPass``

* ``PassManager.run(analysis) -> CompletedPass`` (passe must be an analysis for that signature)
* ``PassManager.run(passe, tree) -> CompletedPass``
* ``PassManager.run(passe, tree, node) -> CompletedPass``

Access to all Tree instances know by the passmanager

* ``PassManager.forest`` is the system-wide collection of ``Tree`` instances,
  refer to the code model abstraction for more informations. 
 

Operations on the forest

* ``PassManager.add(tree)`` is used to add a tree to the system. 
  Two modules cannot have the same identifier nor the same root node identity.
  
  It's not required to explictly add a tree before running a pass if it's provided
  to the ``gather``/``apply``/``run`` method.

  This can also be done from within a forest-wide transformation by doing ::
    connector.apply(add_tree(Tree(...)))

* ``PassManager.remove(tree)`` is used to discard a tree from the system. 
  Potentially to be replaced for one that better fits.

  This can also be done from within a forest-wide transformation by doing ::
    connector.apply(remove_tree(Tree(...)))

### How to declare a pass

There are two kinds of passes: transformation and analysis.

Passes runs on one of these three categories of element: node, tree or forest. 

The generic process to declare a pass is to use a decorator: ``@analysis()`` or ``@transformation()`` 
that wraps a two positional arguments function. 
The first positional argument will be the "connector", it's a bridge to the passmanager that can be 
used from whithin the pass. The second positional argument is the element the passe is run onto
(it can be a node, tree or forest). 

Declare an analysis, use the function decorator: 

```python
def analysis(on, deps=(), cache=True, immutable=False)
```

Examples: 

```python
# a trivial node-wide analysis
@analysis(on='node')
def lsh_to_rsh(_, node):
  yield 'result', {node.targets[0]: node.value}

# if you really want to, the above analysis could be typed as follow: 
@analysis(on='node')
def lsh_to_rsh(_: Connector, node: ast.Assign) -> AR[dict[ast.expr, ast.expr]]: ...

# the node can be type checked at runtime as well with:
@analysis(on=ast.Assign)
def lsh_to_rsh(_, node): ...

# A tree-wide analysis
@analysis(on='tree') # or
@analysis(on=Tree)
def imports(c, tree) -> AR[dict[ast.alias, Import]]:
  r = {}
  for node in ast.walk(tree.root): # this really not opti, but this is demo
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
      continue
    r.update(parse_import(node, modname=tree.identifier))
  trees = c.gather(forest)

  yield 'result', r
  yield 'complete', all(imp.orgmodule in trees for imp in r.values())

 Note that only a forest-wide pass have direct access to the Forest instance. 
 In contrast, a tree-wide pass can request an analysis to be run on another tree
 by passing the tree identifier to the gather() method.

# A forest-wide analysis, that returns the forest to be used in dependent analyses.
@analysis(on='forest')
def forest(_, e): 
  yield 'result', e

# A forest-wide analysis
@analysis(on='forest')
@analysis(on=Forest)
def import_graph(c: Connector, forest: Forest): ...
```

Declare a transformation: 

```python
def transformation(on, deps=())
```

```python
@transformation(on=ast.ClassDef)
def remove_useless_object_base(c, node): ...

@transformation(on=Tree)
def fold_constants(c, node): ...
```

The connector API

```python
class Connector:
  deps: Dependencies # dynamic object
  def gather():... # See PassManager.gather()
  def apply():... # See PassManager.apply()
  def run():... # See PassManager.run()
```

The Pass class API: 

- ``Pass.func``: The wrapped function containing the driving logic the pass.
- ``Pass.on``: the level this pass runs on: FOREST / TREE / NODE
- ``Pass.kind``: the kind of pass: ANALYSIS or TRANSFORMATION

Specifying interactions between passes

- ``Pass.deps``:  The ``PassManager`` can handle the execution of 
  passes for you list dependencies between the various passes. 
  Each pass can declare the set of dependencies in this sequence.  

  - Transformations in the dependencies will be applied before the current pass is run. 
  - Analyses results will be lazily bound to attributes with the corresponding name in the `.deps` attribute of the connector.
  - A pass that requires access to other trees is called an forest-wide pass. 
    This property is transitive and is determined dynamically when the pass runs.

Sometimes it is practical for a pass to have different behaviour depending on optional or required parameters.
Instead of writing several subclass calling an underlying function with different parameters, write an analysis taking parameters: 

To create your parameterized analysis simply call your analysis class with parameters. 
This will create a derived pass with the parameters set.

- ``Pass.params: tuple``: The required parameters of the pass.
- ``Pass.optional_params: frozendict`` The optional parameters of the pass.
- ``Pass.args: frozendict`` The added arguments.
- ``Pass.__call__(*a, **k) -> Pass``: Add argument(s) to the pass.
- ``Pass.clear() -> Pass``: Return this pass without any arguments added.


Running a pass: The ``PassManager`` is the only object responsible of running passes, 
through the ``gather()`` or ``apply()`` methods. 

**The Transformation interface**

- ``Transformation.update``: Instance attribute indicating whether the transformation did something.
- ``Transformation.preservesAnalyses``: Instance attribute listing analyses that are preserved after the transformation updates the content. If a preserved analysis takes parameters, the "like" pattern should be created instead of simply listing the class name (i.e. if the transform preserves the parameterized analysis ``instance_variables`` for any value of its parameter ``inherited``, one must write ``instance_variables.like(inherited=lambda v:True)``, where ``v`` is the value of the parameter)  

**The Analysis interface**

- ``Analysis.doNotCache``: Class attribute indicates to the pass manager not to cache the results of this analysis.
- ``Analysis.isComplete``: Instance attribute indicates to the pass manager to never clear the results when a new module is added to the system.
- ``Analysis.like(**kw)``: Create a like pattern to indicate several parameter values in the context of ``preservesAnalyses``.

**Analysis invalidation**

The analyses invalidated by the current transformation are cleaned up from the cache eagerly. This might not be the smartest thing to do. We might consider marking the results as stale and only clear and update it when a new value is requested. 

**Pass factories**

Some passes are not worth declaring a new ``Pass`` subclass, when a pass has no dependencies for instance; 

Here are some factories to convert various forms of objects into a ``Pass`` type:

- ``Analysis.fromNodeVisitor(visitor)``: Converts a node visitor into an analysis. 
  The pass must not have any requirements. The class must expose a ``visit()`` method and the result be stored in ``self.result`` instance variable. 
- ``Analysis.fromCallable(callable: Callable[[Node], object])``: Converts a one-argument callable into an analysis. The pass must not have any requirements.
- ``Transformation.fromNodeTransformer(transformer)``: Converts a node transformer into a transformation. 
  The pass must not have any requirements. The class must expose a ``visit()`` method and whether the transformation changed the content stored in ``self.update`` instance variable. 

Declaring a ``Pass`` subclass is required if:

- the pass uses parameters or,
- the pass uses optimizations like ``Analysis.isComplete`` or ``Transformation.preservesAnalyses`` or ````Transformation.recAddNode()`` and friends.

**Pass instrumentation**

The pass manager offers a hook system, see `PassManager.hooks.install()`. 

**The module finder**

The finder is in charge of finding python modules and creating ``Module`` instances for them based on a given search context.
It can load modules by fully qualified name or load all modules under a given path. 


## Bundled analyses and transformations

All analyses are made available in the ``libstatic.analyses`` module, all transformations in ``libstatic.transformations`` .

**Normalizations**

Dead code remover: `remove_dead_code`

``__all__`` variable shenanigans normalizations: ``normalize__all__`.

**Node ancestors**

Maps each node to their parents in the syntax-tree.

Analyses: `ancestors`, `ancestor`, 
    `enclosing_scope`, `all_enclosing_scopes`.

**Import resolution**

Analyses: `parsed_imports`, `definitions_of_imports`. 

Transformations: `expand_wildcards`.

**Chains of definitions**

Analyses: `def_use_chains`, `use_def_chains`.

**Function parameters**

Analyses: `function_params`, `function_sugnature`.

**Method resolution order**

Analyses: `method_resolution_order`.

**Attributes**

Analyses: `locals_map`, `ivars_maps`, `get_submodule`, `get_local`, `get_ivar`, `get_attribute`

**Symbolic evaluation**

Analyses: `literal_eval`.

**Reachability**

Analyses: `unreachable_nodes`.

**Type inference**

Analyses: `infer_type`.


Reserve analyses named by the pattern ``get_*`` for the 
ones that take a required parameter. 
So they can be seem as an analysis factory with the mental model: 
  get_local('C') ~> local_C
  get_submodule('exceptions') ~> submodule_exceptions