"""
Code analysis, based on `beniget <https://github.com/serge-sans-paille/beniget>`_.

The core model, provided by ``beniget``, is basically two directed graphs linking definitions to their use and vice versa.
This model is extended in order to include imported names, including the ones from wildcard imports. 
Additionnaly, reachability analysis helps with cutting down the number of potential definitions 
for a given variable, giving more procise results. From there, we can do interesting things, like going 
to the genuine definition of an imported symbol, given the fact that we run python 3.11 for instance.
Or the opposite, finding all references of a given symbol, accross all modules in the prohect.

The `Project` class represent the primary hight-level interface for the library. 
Althought, many other lower-level parts can be used indenpedently.

When creating a `Project`, keep in mind that all module should be added **before** calling `Project.analyze_project`. 
For performance reasons, it does not analyze the use of the builtins or any other dependent modules by default. 
Pass option ``builtins=True`` to analyze builtins or option ``dependencies=True`` to recusively find and 
load any dependent modules (including builtins).


"""
