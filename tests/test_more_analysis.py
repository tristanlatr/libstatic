from libstatic._lib import passmanager
from libstatic._lib.chains import defuse_chains_and_locals, usedef_chains
# real-life analyses

class beniget_analsis(passmanager.ModuleAnalysis):
    def __init__(self):
        self.result = {'chains':None, 'locals':None}
    
    def visit_Module(self, node):
        mod: passmanager.Module = self.passmanager.modules[node]
        modname = mod.modname
        is_package = mod.is_package
        filename = mod.filename
        
        chains, locs, builtins_chains = defuse_chains_and_locals(node, 
            modname=modname, 
            filename=filename, 
            is_package=is_package)
        
        self.result['chains'] = chains
        self.result['locals'] = locs
        return self.result

class def_use_chains(passmanager.ModuleAnalysis):
    
    deps = (beniget_analsis, )

    def __init__(self):
        self.result = None
    
    def visit_Module(self, node):
        self.result = self.beniget_analsis['chains']
        return self.result

class locals_map(passmanager.ModuleAnalysis):
    
    deps = (beniget_analsis, )

    def __init__(self):
        self.result = None
    
    def visit_Module(self, node):
        self.result = self.beniget_analsis['locals']
        return self.result

class use_def_chains(passmanager.ModuleAnalysis):

    deps = (def_use_chains, )

    def __init__(self):
        self.result = None
    
    def visit_Module(self, node):
        return usedef_chains(self.def_use_chains)

# real-life transforms

class remove_dead_code(passmanager.Transformation):

    deps = (def_use_chains, )
    
    def visit_alias(node):
        ...

class remove_unused_imports(passmanager.Transformation):

    deps = (def_use_chains, )
    
    def visit_alias(node):
        ...

class transform_dunder_all_modifications(passmanager.Transformation):

    deps = (def_use_chains, )
    