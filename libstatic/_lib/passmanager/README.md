# ``twisted.python.code`` package proposal

Summary
-------


This new package introduces a framework designed to help work with the abstract syntax tree (AST) analysis and rewriting.
It is an implementation of the pass manager architectural pattern in compiler design - comparable to the ones found in [LLVM](https://llvm.org/docs/WritingAnLLVMPass.html#what-passmanager-does
), [GCC](https://gcc.gnu.org/onlinedocs/gccint/Pass-manager.html) or [pythran](https://pythran.readthedocs.io/en/latest/TUTORIAL.html). 

The pass manager is a component that orchestrates the execution of various passes on the code being processed with the objective of
effective code analysis and transformation in compilers, static analyzers, and related tools.

Each pass declares class attributes that represents everything we need to know about that pass, what dependencies it needs, etc.
The pass manager is responsible to optimize analyses results so they don't have to be run unnecessarily; as well as arranges for the transformations to happen in the correct order and the analyses cache to be invalidated when it needs. 

Since python is a highly dynamic language and one module can affect the state of other modules, 
the pass manager works on an entire system of packages and modules, not just a module at a time. 
Passes that do not require awareness of the other modules in the system are called _intra-module passes_, on the other hand, passes that 
requires access are called _inter-modules passes_. 

Alongside the framework, a collection of qualitative passes and utilities is provided to make development and adoption faster. The exact scope for the 
included passes is still to be determined, and implementation for additional passes and utilities can be contributed in a second phase. 

Package structure: 

```tree
twisted/python/code/
├─ __init__.py
├─ passmanager/
│  ├─ __init__.py
│  ├─ exceptions.py
│  ├─ finder.py
│  ├─ pass manager implementaion modules...
├─ utils.py
├─ analyses/
│  ├─ __init__.py
│  ├─ analysis implementations...
├─ transformations/
│  ├─ __init__.py
│  ├─ transformation implementations...
```

Non-Goals
---------

- Inter-modules refactoring utilities, like renaming a module/class etc. While it could be included, some existing libraries are doing a good
  job for these duties. You can use [rope](https://rope.readthedocs.io/en/latest/library.html#rename) or [libcst](https://libcst.readthedocs.io/). 
  The transformation interface is designed with optimization and normalization in mind, not cross-modules refactoring.
- Introspection of modules. One could write code that generates AST for a given module by inspection (like [astroid](https://github.com/pylint-dev/astroid/blob/v3.2.0/astroid/raw_building.py) does). But this is not in the scope and can easily be done on the client side.
- Some [old twisted ticket](https://github.com/twisted/twisted/issues/4531) has some discussions about including static analysis in ``twisted.python.modules``, I disagree with this approach since the delimitation between introspection and static analysis would become blurry.


Motivation
----------

Why should this work be done? 

Pydoctor needs a framework to work with the standard library ``ast``, particularly to handle the issue of wildcard imports 
and dynamically built ``__all__`` variables; as well as reach deeper understanding of the syntax tree in general ([pydoctor issue #348](https://github.com/twisted/pydoctor/issues/348), [pydoctor issue #591](https://github.com/twisted/pydoctor/issues/591), [pydoctor issue #469](https://github.com/twisted/pydoctor/issues/469)). 
I've been looking for a framework that works with the standard library and would be able to help with these non trivial tasks. It was unsuccessful. See _Alternatives_.

Both pyright and mypy support dynamically built ``__all__`` variables as well as wildcard imports, this makes the python feature 
used across a lot of code, including stub modules. Making the attempt to provide type inference based on typeshed stubs is irrelevant 
for those without the ability to statically resolve wildcards imports (There are currently more than 200 wildcard import in the typeshed stubs).

<!-- 
How does it compare to the competition, if any?

TODO -->

Rationale
---------

Why particular design decisions were made.

- Why a pass manager? 
  The pass manager design allows developers to write transformations depending on arbitrary analyses.
  This feature can be very useful to normalize the AST before passing it to your main visitor. E.g. to
  apply the [undataclass](https://www.pythonmorsels.com/undataclass/) transformation in order to seemlessly 
  support dataclasses by your app since the implicit methods will become explicit; Or expand wildcard imports, which are a major pain for most static analysis tools.

- Setting custom attributes on AST node is a bad practice; it doesn't plays well with AST transformations.
  Instead, analysis results should be stored in other data structures that potentially contains references to nodes.
  This way, when the AST is transformed, the pass manager has two options: 
    - keep the analysis if it's still valid after the transformation - or 
    - clear it from the cache if the transformation invalidated it

- Pass reproducibility is required, meaning (like in [LLVM](https://llvm.org/docs/WritingAnLLVMPass.html)) a pass cannot maintain a state across invocations of ``Pass.doPass()`` (including global data). This restriction is needed in order to keep the cache sain.

- Analysis should be "pure": an analysis cannot manually update other analysis datastructures. This is also because analysis cache must be sain.

- A transformation can only change the node being passed to ``Pass.doPass()`` (which is always a module currently, but it's 
  more a design restriction and might be lifted in the future). If the transformation affects the API of a module (e.g by renaming a class), 
  the dependant modules should also be adjusted with an applicable transformation until no other modules are affected. 

- A transformation can mark an analysis as preserved and manually update the resulting datastructure so it doesn't have to be re-run. 
  This implies that a transformation result can never be cached since it might update other analysis data structures. 
  
- The pass manager implementation is not coupled to any particular tree type, meaning you can use the it with trees generated
  by the [parso](https://parso.readthedocs.io/en/latest/index.html) library; given the fact that the pass manager is provided with the compatible strategy. Nevertheless, implementation of included 
  passes and examples will always use the standard library.

- It is not helpful to consider analyses that runs only on functions or only classes appart from analyses that runs on any kind of nodes. Classes are callable and an expression might refer to a function definition. 
An analysis that runs on modules, in contrast, is different because the pass manager will always run it on the root
module when it's declared as a pass dependency. A move that cannot be done with confidence for other kind of nodes.

Modules are the only thing we can be relatively sure they are what the AST says they are. But even there, with the beauty of Python, one can dynamically set `sys.modules['something']` to an arbitrary object. Like the [klein.resource](https://github.com/twisted/klein/blob/23.12.0/src/klein/resource.py) module which is implemented as an instance.


- When a module is added to the system, the inter-modules analyses not marked as "complete" will be cleared from the cache. 
  This optimization allows for an analysis result to return best-effort results when information is missing in the system and later get the complete result after the required module is added to the system. 

- When a module is removed, all inter-modules analyses are cleared no matter their completeness. 

Specification
-------------

**The module model**

Before running a pass, one must add the module to the pass management system.

```python
class Module:
    node: ast.Module
    modname: str
    filename: str | None = None
    isPackage: bool = False
    isStub: bool = False
    code: str | None = None
```

Modules can be created manually or with the finder (discussed below).

**The PassManager interface**

* ``PassManager.addModule(module)`` is used to add a module to the system, and is required before running a pass on that module. Two modules cannot have the same name nor the same ast node identity.
* ``PassManager.removeModule(module)`` is used to discard a module from the system. Potentially to be replaced for one that better fits.
* ``PassManager.gather(analysis, node) -> object`` is used to gather (!) the result of an analysis on an AST node.
* ``PassManager.apply(transformation, node) -> tuple[bool, node]`` is used to apply (sic) a transformation on an AST node. Returns whether the pass updated the content and the modified node.
* ``PassManager.modules`` is a mapping to access the collection of added ``Module``s. Modules can be retrieved by module qualified name or by node (any node in the module). The  pass manager needs to maintain a map of all nodes in the system to their root module, this is required for the cache to know which module entry to use when given arbitrary nodes. See the transformation optimizations to reduce the cost of this operation.
* ``PassManager.support(ast)`` is used to switch the pass manager AST support kind. Valid values includes ``gast`` or ``ast``. 

**The Pass interface**

There are two kinds of passes: transformation and analysis.
    
* ``ModuleAnalysis`` and ``NodeAnalysis`` are to be
    subclassed by any pass that collects information about the AST.
* ``Transformation`` is to be sub-classed by any pass that updates the AST.

Pass class hierarchy:

```
Pass
├─ Analysis
├─ ├─ ModuleAnalysis
├─ ├─ NodeAnalysis
├─ Transformation
```

Specifying interactions between passes

- ``Pass.dependencies``:  The ``PassManager`` tries to optimize the execution of passes it must know how the passes interact with each other and what dependencies exist between the various passes. 
  Each pass should declare the set of dependencies in this sequence.  

  - Transformations in the dependencies will be applied before the current pass is run. 
  - Analyses results will be lazily bound to instances variables with the corresponding class name.
  - A pass that requires access to other modules is called an inter-modules pass. This property is transitive. To get access to other modules one need to 
  list the special ``passmanager.modules`` analysis in it's dependencies. Modules will then be accessible throught ``self.modules``. 


Sometimes it is practical for a pass to have different behaviour depending on optional or required parameters.
Instead of writing several subclass calling an underlying function with different parameters, write an analysis taking parameters: 

- ``Pass.requiredParameters``: Class variable tuple, all required parameters names.
- ``Pass.optionalParameters``: Class variable dict, keys are parameters names are values are the default values in case none is provided.

To create your parameterized analysis simply call your analysis class with keywords corresponding to your parameters. This will create a derived subclass with the parameters set at class level, it only creates a instances if called without any parameters. This is implemented with a metaclass that overrides ``__call__``.

- ``Pass.prepareClass``: A classmethod that is called when the subclass is created, it can be used to dynamically adjust the dependencies based on provided parameters.

Running a pass: The ``PassManager`` is the only object that should instantiate passes, through the ``gather()`` or ``apply()`` methods. These methods creates the pass instance, set up the variables and run the pass. 

- ``Pass.passmanager``: Instance variable, ref to the pass manager.
- ``Pass.ctx``: Instance variable tracking the pass context: ``self.ctx.module`` is the current ``Module`` and ``self.ctx.current`` is the ast node the current pass is running on.
- ``Pass.doPass(ndoe) -> object``: Abstract method that needs to be implemented by all concrete Pass types: this is where the core of the pass logic goes.
- ``Pass.run(ndoe) -> object``: Method to run the pass on the given node. This methods calls ``Pass.prepare()`` then ``Pass.doPass``. Override this method to provide pre or post processing to the pass.

**The Transformation interface**

- ``Transformation.update``: Instance attribute indicating whether the transformation did something.
- ``Transformation.preservesAnalyses``: Instance attribute listing analyses that are preserved after the transformation updates the content. If a preserved analysis takes parameters, the "like" pattern should be created instead of simply listing the class name (i.e. if the transform preserves the parameterized analysis ``instance_variables`` for any value of its parameter ``inherited``, one must write ``instance_variables.like(inherited=lambda v:True)``, where ``v`` is the value of the parameter)  
- ``Transformation.addedNodes``: Instance attribute listing the new nodes added to the tree, optionally filled. These are used to optimize post transformation tree visiting. 
- ``Transformation.removedNodes``: Instance attribute listing the new nodes added to the tree, optionally filled. 

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
- ``Analysis.fromSimpleCallable(callable: Callable[[Node], object])``: Converts a one-argument callable into an analysis. The pass must not have any requirements.
- ``Analysis.fromCallable(callable: Callable[[PassManager, Node], object], isInterModules=False)``: Converts a two-argument callable where the first argument is the pass manager and the second the node. The analysis can use ``gather`` method to request requirements. If the parameter ``isInterModules`` is true, the analysis will be allowed to depend on other inter-modules analyses and request the ``passmanger.modules`` analysis.
- ``Transformation.fromNodeTransformer(transformer)``: Converts a node transformer into a transformation. 
  The pass must not have any requirements. The class must expose a ``visit()`` method and whether the transformation changed the content stored in ``self.update`` instance variable. 

Declaring a ``Pass`` subclass is required if:

- the pass uses parameters or,
- the pass uses optimizations like ``Analysis.isComplete`` or ``Transformation.preservesAnalyses``.

**Pass instrumentation**

The pass manager uses an event dispatcher with the following event definition; some of them are designed only to be 
used by client code, some are used by default. To add a custom event listener, use ``PassManger.dispatcher.addEventListener``. 

```
Event
├─ ModuleTransformedEvent
├─ ClearAnalysisEvent
├─ ModuleAddedEvent
├─ ModuleRemovedEvent
├─ RunningTransform
├─ TransformEnded
├─ RunningAnalysis
├─ AnalysisEnded
├─ SupportLibraryEvent
```

**The module finder**

The finder is in charge of finding python modules and creating ``Module`` instances for them based on a given search context.
It can load modules by fully qualified name or load all modules under a given path. 

Future developments
--------------------

- File based caching; this requires generating a hash of the AST in order to cache 
  intra-modules analyses. For inter-modules analyses, it will requires a inter-modules hash. Some [similar work](https://github.com/QuantStack/memestra/blob/0.2.1/memestra/caching.py) has already been done. 
- Provide a pipeline API: Pass managers might organize passes into pipelines, but this will be left to client code for the moment.
- Provide support for namespace packages
- Having transformations that run only on function definitions, makes sens. Because we could make more cache optimizations as long as the function is pure and its API is not changed, typically opmization passes: it must not affect the global scope - before and/or after the transform.


Alternatives
------------

- ``rope``: Rope runs static analysis on the whole project at once when requested. This task can be very time consuming.
  The approach here, in opposit, is to provide a framework to optimize passes and run only the minimum required.
- ``astroid``: While a great library, it does not offer the flexibility we thrive for - it includes a couples of APIs the main
  one being ``NodeNG.infer()`` that returns a generator for potential values. I've tried to use it but could not make it evaluate
  the values of ``__all__`` variables without patching astroid internals (which is not really inline with the GNU Lesser General Public License v2.1). Also it does not work with the standard library AST.
- ``pytype``: There is an included library in pytype: [traces](https://google.github.io/pytype/developers/tools.html#traces), but pytype analyzer is not known to be fast and working with opcode traces adds a layer of complexity. 
- ``mypy``: Some [code](https://github.com/pyastrx/pyastrx/blob/v0.5.2/pyastrx/inference/mypy.py) in the ``pyastrx`` project shows that we can use ``mypy`` to extract type inference results and later matches these to node instances. But this approach does not offer enough flexibility and it is rather "hacky". 

Reference Implementation
------------------------

The implementation is currently in development and misses a lot of optimization features and unit tests. 

But here is where you can look at the code:

https://github.com/tristanlatr/libstatic/tree/passmanager/libstatic/_lib/passmanager

Dependencies
------------

- The pass manager infrastructure itself does not have any external dependencies.  

- The finder module uses ``typeshed_client``.

- The def-use chains analysis is provided by [beniget](https://github.com/serge-sans-paille/beniget), so many interesting analyses will require it;
  Including the evaluation of ``__all__`` variables, type inference and others.

Usages
------

- Support ``pydoctor`` in its code analysis.

Thanks
------

Special thanks to [@serge-sans-paille](https://github.com/serge-sans-paille) for all the help and reviews.
