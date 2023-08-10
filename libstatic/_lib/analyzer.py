import ast
from itertools import chain

from typing import Any, Dict, Set, Union, cast

from ..model import MutableState, NameDef, Options, Mod, Def
from .chains import defuse_chains_and_locals, usedef_chains, BuiltinsChains
from .ancestors import Ancestors
from .reachability import get_unreachable
from .wildcards import compute_wildcards
from ..exceptions import StaticException
from .shared import StmtVisitor

from beniget.beniget import BuiltinsSrc # type: ignore

class ChainDefUseOfImports(StmtVisitor):
    """
    Adds each alias instance to the list of uses of the Def of the name their are binding.
    """

    # TODO: visit_Module should return Mapping[Def, List[Def]]
    # and 'State' should be used here.
    def __init__(self, state: "MutableState") -> None:
        self._state = state

    def generic_visit(self, node: Any) -> None:
        if isinstance(node, ast.expr):
            return
        else:
            super().generic_visit(node)

    def visit_Import(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        for alias in node.names:
            alias_def = self._state.get_def(alias)
            orgmodule = self._state.get_module(alias_def.orgmodule)
            if orgmodule:
                orgname = alias_def.orgname
                if orgname:
                    if orgname == "*":
                        continue
                    try:
                        defs = self._state.get_attribute(
                            orgmodule, orgname
                        )  # todo: handle ignore locals
                    except StaticException:
                        continue
                        # import target not found
                    else:
                        for loc in defs:
                            self._state.add_user(loc, alias_def)
                else:
                    self._state.add_user(orgmodule, alias_def)
            else:
                # module not found in the system
                continue

    visit_ImportFrom = visit_Import

class Analyzer:
    _recurse_up_to = 8

    def __init__(self, state: MutableState, options: Options) -> None:
        self._options = options
        self._state = state
        self._builtins_dict: Dict[str, 'Def'] = {}

    def _analyze_builtin_module(self, mod:Mod) -> None:
        # the builtins module should always be processed first since all other modules
        # implicitly depends on it and def-use chains for builtins must be computed once we
        # have the defs in the builtins module of course.
        
        # - create the builtins mapping
        
        builtins_self_chains = self._analyze_module_pass1(mod)

        # use reachability analysis to only accounts for symbols 
        # available for the right python version.
        for d in filter(lambda l: l is not None and self._state.is_reachable(l), 
                        chain.from_iterable(self._state.get_locals(mod).values())):
            # Mypy is not smart enought yet to narrow the type of 'd' to NameDef.
            name = cast(NameDef, d).name()
            # we can't trust the islive flag here, because
            # https://github.com/serge-sans-paille/beniget/pull/73
            if name in BuiltinsSrc:
                self._builtins_dict[name] = cast(NameDef, d)
                # If we have two (or more) definitions for this builtin name,
                # we under-approximate by taking the last defined name.
        
        if len(self._builtins_dict) < len(BuiltinsSrc):
            self._state.msg(f'missing builtin names: {sorted(set(BuiltinsSrc).difference(self._builtins_dict))}', thresh=1)

        self._link_builtins_chains(builtins_self_chains)

    def _link_builtins_chains(self, builtins_defuse: BuiltinsChains) -> None:
        # If the builtins module is not in the system, this is a no-op.

        # TODO: Does this needs to handle KeyError?
        for name in self._builtins_dict:
            definition = builtins_defuse[name]
            try:
                next(iter(definition.users()))
            except StopIteration:
                continue
            proper_builtin_def = self._builtins_dict[name]
            for u in definition.users():
                self._state.add_user(proper_builtin_def, u)

    def _analyze_module_pass1(self, mod: "Mod") -> BuiltinsChains:
        # Returns the builtins chains of the module, the value is 
        # used to process the builtins usages inside the module.

        self._state.msg(f'analyzing {mod.name()}', thresh=1)
        module_node = mod.node

        # - compute ancestors, first thing to do because it's used
        # to fetch the filename for all ast nodes, which might be useful
        # for error/warning reporting.
        ancestors_vis = Ancestors()
        ancestors_vis.visit(mod.node)
        ancestors = ancestors_vis.parents
        self._state.store_anaysis(ancestors=ancestors)

        # : Accumulate static analysis infos from beniget
        # - compute local def-use chains
        # - parsing imports
        defuse, locals, builtins_defuse = defuse_chains_and_locals(
            module_node,
            filename=mod.filename(),
            modname=mod.name(),
            is_package=mod.is_package,
        )
        self._state.store_anaysis(defuse=defuse, locals=locals)
        
        # : Flip def def-use and generate the use-def chains
        usedef = usedef_chains(defuse)
        self._state.store_anaysis(usedef=usedef)

        # : Reachability analysis
        unreachable = get_unreachable(self._state, self._options, module_node)
        self._state.store_anaysis(unreachable=unreachable)
        
        return builtins_defuse

    def _analyzer_pass1(self) -> None:
        
        not_found = set()
        to_process = [mod.name() for mod in self._state.get_all_modules()]
        processed_modules: Set[str] = set()
        max_iterations = 1 if not self._options.dependencies else (
            self._recurse_up_to+1 if isinstance(self._options.dependencies, bool) 
            else int(self._options.dependencies)+1)
        iteration = 0

        # builtins module should be processed first, otherwise insertion order
        to_process.sort(key=lambda e:e!='builtins')

        while to_process:
            iteration += 1
            for name in list(to_process):
                to_process.remove(name)
                if name not in processed_modules:
                    # load dependency modules if dependencies is True and we 
                    # haven't reached the maximum number of iterations
                    mod = self._state.get_module(name)
                    if mod is None and name not in not_found:
                        mod = self._state.add_typeshed_module(name)
                    if mod is None:
                        not_found.add(name)
                        continue
                    
                    if name == 'builtins':
                        assert iteration == 1, 'unexpected builtins module'
                        self._analyze_builtin_module(mod)
                    else:
                        self._link_builtins_chains(
                            self._analyze_module_pass1(mod))

                    # collect dependencies
                    # TODO: this could use some fine tuning
                    deps = [self._state.get_def(al).orgmodule
                        for al in (
                            n
                            for n in ast.walk(mod.node)
                            if isinstance(n, ast.alias)
                        )]
                    deps = [d for d in deps if d not in chain(processed_modules, to_process)]

                    # add dependency modules names to the process list.
                    
                    if deps:
                        if iteration != max_iterations:
                            self._state.msg(f'collected {len(deps)} dependencies from module {mod.name()}', thresh=1)
                            to_process.extend(deps)
                        elif max_iterations>1:
                            self._state.msg(f'maximum number of iterations reached, skipping {len(deps)} dependencies from module {mod.name()}: {", ".join(deps)}')

                processed_modules.add(name)
    
    def _analyzer_pass2(self) -> None:
        # : Imports analysis: complement def-use chains with import chains
        # must be done after all modules have been analyzed 
        for mod in self._state.get_all_modules():
            ChainDefUseOfImports(self._state).visit(mod.node)
    
    def _analyzer_pass3(self) -> None:
        # at this point goto definition is working, for non wildcard imports names
        # : Compute __all__ and wildcard imports: fixup def-use chains for wildcard imports,
        # the compute_wildcards() function will mutate the state chains to resolve all widlcard
        # imported names.
        self._state._dunder_all = compute_wildcards(self._state)
    
    def analyze(self) -> None:
        """
        Initiate the project state.
        """

        if ((self._options.dependencies or self._options.builtins)
            and 'builtins' not in self._state._modules):
            if not self._state.add_typeshed_module('builtins'):
                raise RuntimeError('missing the builtins module :/')
                
        self._analyzer_pass1()
        self._analyzer_pass2()
        self._analyzer_pass3()

        
        
        
