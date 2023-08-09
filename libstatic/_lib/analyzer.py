import ast
from itertools import chain

from typing import Any, Dict, Union

from ..model import MutableState, Options, Mod, Def
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
    def __init__(self, state: MutableState, options: Options) -> None:
        self._options = options
        self._state = state
        self._builtins_dict: Dict[str, 'Def'] = {}

    def _compute_ancestors(self, mod:'Mod') -> None:
        # - compute ancestors
        ancestors_vis = Ancestors()
        ancestors_vis.visit(mod.node)
        ancestors = ancestors_vis.parents
        self._state.store_anaysis(ancestors=ancestors)
    
    def _init_builtin_module(self) -> None:
        # the builtins module should always be processed first since all other modules
        # implicitly depends on it and def-use chains for builtins must be computed once we
        # have the defs in the builtins module of course.
        
        # - create the builtins mapping
        
        builtins = self._state.get_module('builtins')
        
        if not builtins:
            return None
        
        self_chains = self._analyze_module_pass1(builtins)

        for d in filter(None, chain.from_iterable(
            self._state.get_locals(builtins).values())):
            name = d.name()
            # we can't trust the islive flag here, but why?
            if name in BuiltinsSrc:
                self._builtins_dict[name] = d
                # If we have two (or more) definitions for this builtin name,
                # we under-approximate by taking the last defined name.
                # TODO: do better here, use reachability analysis
        
        if len(self._builtins_dict) < len(BuiltinsSrc):
            self._state.msg(f'missing builtin names: {sorted(set(BuiltinsSrc).difference(self._builtins_dict))}', thresh=1)

        self._link_builtins_chains(self_chains)

    def _link_builtins_chains(self, builtins_defuse: BuiltinsChains) -> None:
        # If the buitlins module is not in the system, this is a no-op.

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

        # - compute ancestors
        self._compute_ancestors(mod)

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

        usedef = usedef_chains(defuse)
        self._state.store_anaysis(usedef=usedef)

        # : Reachability analysis
        unreachable = get_unreachable(self._state, self._options, module_node)
        self._state.store_anaysis(unreachable=unreachable)
        
        return builtins_defuse

    def _analyzer_pass1(self) -> None:
        processed_modules = set()
        not_found = set()
        to_process = [mod.name() for mod in self._state.get_all_modules()]
        iteration = 0

        while to_process:
            for name in list(to_process):
                to_process.remove(name)
                if name not in processed_modules:
                    # load dependency modules if nested_dependencies is not zero
                    mod = self._state.get_module(name)
                    if mod is None and name not in not_found:
                        mod = self._state.add_typeshed_module(name)
                    if mod is None:
                        not_found.add(name)
                        continue

                    if name != 'builtins':
                        # we handle the builtins module analysis elsewhere.
                        b = self._analyze_module_pass1(mod)
                        self._link_builtins_chains(b)

                    # add dependencies
                    if iteration != self._options.nested_dependencies:
                        # TODO: this could use some fine tuning
                        deps = [self._state.get_def(al).orgmodule
                            for al in (
                                n
                                for n in ast.walk(mod.node)
                                if isinstance(n, ast.alias)
                            )]
                        deps = [d for d in deps if d not in processed_modules]
                        
                        self._state.msg(f'collected {len(deps)} dependencies from module {mod.name()}', thresh=1)
                        to_process.extend(deps)
                processed_modules.add(name)
            iteration += 1

    def analyze(self) -> None:
        """
        Initiate the project state.
        """
        self._init_builtin_module()
        self._analyzer_pass1()

        # after all modules have been analyzed once, we convert the builtins
        
        # : Imports analysis: complement def-use chains with import chains
        # must be done after all modules have been added
        for mod in self._state.get_all_modules():
            ChainDefUseOfImports(self._state).visit(mod.node)
        
        # at this point goto definition is working, for non wildcard imports names
        # : Compute __all__ and wildcard imports and fixup def-use chains for wildcard imports
        self._state._dunder_all = compute_wildcards(self._state)
