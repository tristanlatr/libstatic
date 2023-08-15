"""
Code analysis, based on `beniget <https://github.com/serge-sans-paille/beniget>`_.

The core model, provided by ``beniget``, is basically two directed graphs linking definitions to their use and vice versa.
This model is extended in order to include imported names, including the ones from wildcard imports. 
Additionnaly, reachability analysis helps with cutting down the number of potential definitions 
for a given variable, giving more precise results. From there, we can do interesting things, like going 
to the genuine definition of an imported symbol, given the fact that we run python 3.11 for instance.
Or the opposite, finding all references of a given symbol, accross all modules in the project.

The `Project` and `State` classes represent the primary hight-level interface for the library. 
Althought, many other lower-level parts can be used indenpedently.

When creating a `Project`, keep in mind that all module should be added **before** calling `Project.analyze_project`. 
For performance reasons, it does not analyze the use of the builtins or any other dependent modules by default. 
Pass option ``builtins=True`` to analyze builtins or option ``dependencies=True`` to recusively find and 
load any dependent modules (including builtins).

Once the project has been analyzed, one can use the `Project.state` attribute, which is the `State`. 
This object present accessors for several kind of analyses.
"""

from ._analyzer.state import Project, State, MutableState, Options
from ._analyzer.loader import load_path
from ._lib.model import (Def, Mod, Cls, Func, Var, Arg, Imp, Comp, Lamb, Attr, 
                         ClosedScope, OpenScope, Scope, NameDef)
from ._lib.process import TopologicalProcessor
from ._lib.exceptions import *

__all__ = (

    "Project", 
    "Options", 
    "State", 
    "MutableState", 
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