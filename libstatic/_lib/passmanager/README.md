# ``twisted.python.code`` package proposal

Summary
-------

This new package introduces a framework designed to help work with the abstract syntax tree (AST) analysis and rewriting.
It is an implementation of the pass manager architectural pattern in compiler design - comparable as the ones found in LLVM, gcc or pythran. 

The pass manager is a component that orchestrates the execution of various passes on the code being processed with the objective of
effective code analysis and transformation in compilers, static analyzers, and related tools.

Each pass delcares class attributes that represents everything we need to know about that pass, what dependencies it needs, etc.
The pass manager is responsible to optimize analyses results so they don't have to be run unecessarly; as well as arranges for the transformations to happen in the correct order and the analyses cache to be invalidated when it needs. 

Since python is a highly dynamic language and one module can affect the state of other modules, 
the passmanager works on an entire system of packages and modules, not just a AST module at a time.

Alongside the framework, a collection of qualititive passes and utlilities is provided to make development and adoption faster. The exact scope for the 
included passes is still to be determined, and implementation for additional passes can be contributed in a second phase. 

Package structure: 

```tree
twisted/python/code/
├─ __init__.py
├─ passmanager/
│  ├─ __init__.py
│  ├─ events.py
│  ├─ exceptions.py
├─ finder/
│  ├─ __init__.py
├─ analyses/
│  ├─ __init__.py
│  ├─ analysis implementations...
├─ transformations/
│  ├─ __init__.py
│  ├─ transformation implementations...
```

Non-Goals
---------

- Iter-modules refactoring utilities, like renaming a module/class etc. 
  You can use [rope](https://rope.readthedocs.io/en/latest/library.html#rename) or [libcst] for that.
- Introspection of modules. One could write code that generates AST for a given module by inspection (like [astroid] does). But this is not in the scope.
- Some [old twisted ticket](https://github.com/twisted/twisted/issues/4531) has some discussions about including static analysis in ``twisted.python.modules``, I disagree with this approach since the delimitation between introspection and static analysis would become blury.


Motivation
----------

Why should this work be done? 

Pydoctor needs a framework to work with the stabdard library ``ast``, particularly to handle the issue of wildcard imports 
and dynamically built ``__all__`` variables; as well as reach deeper understanding of the syntax tree in general. 
I've been looking for a framework that works with the standard library and would be able to help with these non trivial tasks, unsuccessfuly. See _Alternatives_.

Both pyright and mypy support dynamically built ``__all__`` variables as well as wildcard imports, this makes the python feature 
used accross a lot of code, including stub modules. Making the attempt to provide type inference based on typeshed stubs is irrelevant 
for those without the ability to statically resolve wildcards imports.

How does it compare to the competition, if any?

TODO

Rationale
---------

Why particular design decisions were made.

- Why a pass manager? 
  The passmanager design allows developpers to write transformations depending on arbitrary analyses.
  This feature can be very useful to normalize the ast before passing it to your main visitor. E.g. to
  apply the [undataclass](https://www.pythonmorsels.com/undataclass/) transformation in order to seemlessly 
  support dataclasses by your app since the implicit methods will become explicit; Or expand wildcard imports, which are a major pain for most static analysis tools.

- Setting custom attributes on ast node is a bad practice; it doesn't plays well with ast transformations.
  Instead, analysis results should be stored in other datastructures that potentially contains references to nodes.
  This way, when we transform the ast we have two options: 
    - keep the analysis if it's still valid after the transformation - or 
    - clear it from the cache if the transformation invalidated it

- Pass reproducibility is required, this means (like in [LLVM](https://llvm.org/docs/WritingAnLLVMPass.html)), a pass cannot maintain a state across invocations of ``run()`` (including global data). This restriction is needed in order to keep the cache sain.

- Analysis should be "pure": an analysis cannot manually update other analysis datastructures. This is also because analysis cache must be sain.

- A transformation can only change the node beeing passed to ``run()`` (which is always a module currently, but that's 
  more a design restriction and might be lifted in the future). If the transformation affects the api of a module (e.g by renaming a class), 
  the dependant modules should be adjusted as well with an applicable transformation until no other modules are affected. 

- A transformation can mark an analysis as preserved and manually update the resulting datastructure so it doesn't have to be re-run. 
  This implies that an transformation result can never be cached since it might update other analysis datastructures. 
  
- The passmanager implementation is not coupled to any particular de tree types, meaning you can use the passmanager with trees generated
  by the [parso] library; given the fact that the passmanager is provided with the compatible strategy. Nevertheless, implementation of included 
  passes and examples will always use the standard library.

- It is not helpful to consider analyses that runs only on functions or only classes appart from analyses that runs on any kind of nodes. Classes can be callables and an expresion might refer to a function definition. 
An analysis that runs on modules, in constrast, is different because the passmanager will always run it on the root
module when it's declared as a pass dependency. A move that cannot be done with confidence for other kind of nodes.

Modules are the only thing we can be relatively sure they are what the AST say they are at runtime. But even there, with the beauty of Python one can dynamically set `sys.modules['something']` to an arbitrary object. Like the [klein.resource] module which is implemented as a class instance.

So there are only two kind of

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

* ``PassManager.addModule(module)`` is used to add a module to the system, and is required before running a pass on that module. Two modules cannot have the same name, not the same ast node identity.
* ``PassManager.removeModule(module)`` is used to discard a module from the system, potentially to be replaced for one that better fit the use case.
* ``PassManager.gather(analysis, node)`` is used to gather (!) the result of an analyses on an AST node.
* ``PassManager.apply(transformation, node)`` is used to apply (sic) a transformation on an AST node.
* ``PassManager.modules`` is a mapping to access the collection of added ``Module``s. Modules can be retreived by modname or by node.

**The Pass declaration interface**

There are two kinds of passes: transformations and analysis.
    
* ``ModuleAnalysis`` and ``NodeAnalysis`` are to be
    subclassed by any pass that collects information about the AST.
* ``Transformation`` is to be sub-classed by any pass that updates the AST.

Specifying interactions between passes

The ``PassManager`` tries to optimize the execution of passes it must know how the passes interact with each other 
and what dependencies exist between the various passes. 

To track this, each pass can declare the set of dependencies in the ``Pass.dependencies`` sequence.  

- Transformations in the dependencies will be applied before the current pass is run. 
- Analyses results will be lazily bound to instances variables with the corresponding class name.
- A pass that requires access to other modules is called an inter-modules pass. This property is transitive. To get access to other modules one need to 
list ``passmanager.modules`` analysis in it's dependencies. Modules will then be accessible throught ``self.modules``. 

**The Transformation declaration interface**

It is possible to list analyses that are preserved when a transformation is run.  
To track this, declare the set of valid analysis in the  ``Transformation.preserves_analysis`` sequence. 

**The Analysis declaration interface**

- ``Analysis.do_not_cache``: Class attribute indicates to the pass manager not to cache the results of this analysis.

**Analysis invalidation**

TODO: The analyses invalidated by the current transformation are cleaned up from the cache eagerly.

Optimizations of the cache...

**The module finder**

TODO

Future developements
--------------------

- File based caching; this requires generating a hash of the ast in order to cache 
  intra-modules analyses. For inter-modules analyses, it will requires a inter-modules hash. Some [similar work](https://github.com/QuantStack/memestra/blob/0.2.1/memestra/caching.py) has already been done. 
- Provide a pipeline API: Pass managers might organize passes into pipelines, but this will be left to client code for the moment.
- Provide support for namespace packages
- Having transformations that run only on function definitions, makes sens. Because we could make more cache optimizations as long as the function is pure and it's API is not changed, typically opmization passes: it must not affect the global scope - before and/or after the transform.


Alternatives
------------

- rope: Rope runs static analysis on the whole project at once when requested. This task can be very time consuming.
  The approach here, by contrast, is to provide a framework to optimize passes and run only the minimum required.
- astroid: While a great library, it does not offer the flexibility we thrive for - it includes a couples of APIs the main
  one being ``NodeNG.infer()`` that returns a generator for potential values. I've tried to use it but could not make it evaluate
  the values of ``__all__`` variables without patching astroid internals (which is not really inline with the GNU Lesser General Public License v2.1). Also it does not work with the standard library ast.


Reference Implementation
------------------------



Risks and Assumptions
---------------------



Rejected Ideas
--------------



Dependencies
------------



Usages
------

- Support ``pydoctor`` in it's code analysis.
- Replace ``pythran``'s passmanager. 

Thanks
------

Special thanks to [@serge-sans-paille](https://github.com/serge-sans-paille) for all the help and reviews.

Resources
---------

https://pythran.readthedocs.io/en/latest/TUTORIAL.html

https://llvm.org/docs/WritingAnLLVMPass.html#what-passmanager-does

https://gcc.gnu.org/onlinedocs/gccint/Pass-manager.html
