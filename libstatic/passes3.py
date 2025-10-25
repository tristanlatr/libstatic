# TODO: All of the implementations should go into submodules and then exported







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
