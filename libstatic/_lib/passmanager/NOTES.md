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
