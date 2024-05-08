"""
Simple static analysis library for Python, based on `beniget <https://github.com/serge-sans-paille/beniget>`_.

Goals and non-goals
===================

The main goal of this project is to provide a simple, standard library compatible framework to statically analyse a 
collection of related python modules. The initial intent beeing to support static analyzers and API document generators 
working with the `ast`.

Trade-offs
----------

Libstatic tries to be relatively lightweight and fast, so here are some trade-offs:

- Only provide intra-procedural analyses.
- Partial path sensitivity: libstatic relies on over-approximations, some unreachable 
  execution paths will be filtered out but impossibe paths might still be considered.
- No pointer or shape analysis: Aliasing that happens in non-trivial ways will not be detected.
- No soundness guarantees: ignores the effects of `eval`-like, `setattr`, etc. functions on the program state. 
  It doesn’t make worst-case sound assumptions, but rather "reasonable" ones.
- Incomplete type system: While *basic* type inference is provided, 
  libstatic does not carry the complexity to support full-featured type-checking.
  
The model
=========

The core model, provided by ``beniget``, is basically two directed graphs linking definitions to their uses and vice versa.
We call these data structures Def-Use chains and Use-Def chains.
This model is extended in order to include imported names, including the ones from wildcard imports. 

All ast nodes categorized as a use or a definition have a coresponding `Def` instance. Definitions are represented
using one of the specialized `Def` subclass: `Mod`, `Cls`, `Func`, `Var`, `etc <classIndex.html#libstatic.Def>`_... The direct users of a definition
are accessible with `Def.users()`, which returns a collection of `Def` (generally wrapping a `ast.Name` or `ast.alias`).

Additionnaly, reachability analysis helps with cutting down the number of potential definitions 
for a given user, giving more precise results. From there, we can trace the genuine definition 
of any symbol if it's in the system. As well find all references of a given symbol, accross all modules in the project.

The Def-Use chains and Use-Def chains, and other analyses results are made available througth the `State`.

How to use the library
======================

The `Project` and `State` classes represent the primary hight-level interface for the library,
(some other lower-level parts can be used indenpedently). 
The API is designed to work with current code using the standard `ast` module.

- First, create a `Project` instance
- Then add the modules you want to analyze with `Project.add_module` or `load_path`
- Call `Project.analyze_project()`
- Do stuff with `Project.state`

Keep in mind that all module should be added **before** calling ``analyze_project()``. 
For performance reasons, it does not analyze the use of the builtins or any other dependent modules by default. 
Use ``Project(builtins=True)`` to analyze builtins or ``Project(dependencies=True)`` to recusively find and 
load any dependent modules (including builtins), see `Options` for other arguments.

The `State` instance acts like a `façade <https://en.wikipedia.org/wiki/Facade_pattern>`_ and present accessors 
for several kind of analyses.

"""

from ._analyzer.state import Project, State, MutableState, Options
from ._analyzer.loader import load_path
from ._lib.model import (Def, Mod, Cls, Func, Var, Arg, Imp, Comp, Lamb, Attr, 
                         ClosedScope, OpenScope, Scope, NameDef, Type)
from ._lib.process import TopologicalProcessor
from ._lib.exceptions import *

__all__ = (

    "Project", 
    "Options", 
    "State", 
    "MutableState", # not public API
    "load_path",

    "Def",
    "Mod",
    "Cls",
    "Func",
    "Var",
    "Arg", 
    "Imp",
    "Scope",
    "Comp",
    "Lamb",
    "Attr",
    "ClosedScope",
    "OpenScope",
    "NameDef",

    "TopologicalProcessor",

    "Type",

    "NodeLocation",
    "StaticException",
    "StaticNameError",
    "StaticAttributeError",
    "StaticTypeError",
    "StaticImportError",
    "StaticValueError",
    "StaticStateIncomplete",
    "StaticCodeUnsupported",
    "StaticAmbiguity",
    "StaticEvaluationError", 
    "StaticUnknownValue",
)