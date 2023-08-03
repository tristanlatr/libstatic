import ast

from typing import Any, Dict, Mapping, Optional, Union

from ..model import MutableState, Options, Mod, Def
from .chains import defuse_chains_and_locals, usedef_chains, DefUseChains, ast_to_gast
from .ancestors import Ancestors
from .reachability import get_unreachable
from .wildcards import compute_wildcards
from ..exceptions import StaticException
from .shared import StmtVisitor
from .imports import ParseImportedNames
from .chains import DefUseChains, BenigetConverterBuiltins

import gast # type: ignore
from beniget.beniget import Def as BenigetDef # type: ignore

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

        self._builtins_converter: Optional[BenigetConverterBuiltins] = None
        self._converted: Dict[BenigetDef, Optional['Def']] = {}

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
        self._state.msg(f'analyzing the builtins module', thresh=1)

        self._compute_ancestors(builtins)
        
        # def-use
        gastnode, gast2ast = ast_to_gast(builtins.node)
        # if we'a analyzing the builtins module, we can't 
        # use it to map the builtins usages yet.
        duc = DefUseChains(self._state.get_filename(builtins.node))
        setattr(duc, "future_annotations", True)
        duc.visit(gastnode)
        
        # parse imports
        alias2importinfo = ParseImportedNames('builtins', is_package=False).visit_Module(
            builtins.node
        )

        self._builtins_converter = BenigetConverterBuiltins(gast2ast, alias2importinfo, 
                                                            duc, self._converted)

    def _finalize_builtin_module(self):
        # basically connecting all uses of builtins into the converted module now
        builtins = self._state.get_module('builtins')
        if not builtins:
            return
    
        # convert result into standard library
        converter = self._builtins_converter
        defuse, locals = converter.convert(converter.duc)
        self._state.store_anaysis(defuse=defuse, locals=locals)
        
        usedef = usedef_chains(defuse)
        self._state.store_anaysis(usedef=usedef)

        # : Reachability analysis
        unreachable = get_unreachable(self._state, self._options, builtins.node)
        self._state.store_anaysis(unreachable=unreachable)
    
    def _analyze_module_pass1(self, mod: "Mod") -> None:
        self._state.msg(f'analyzing {mod.name()}', thresh=1)
        module_node = mod.node

        # - compute ancestors
        self._compute_ancestors(mod)

        # : Accumulate static analysis infos from beniget
        # - compute local def-use chains
        # - parsing imports
        defuse, locals, converted_defs = defuse_chains_and_locals(
            module_node,
            filename=mod.filename(),
            modname=mod.name(),
            is_package=mod.is_package,
            builtins=self._builtins_converter.builtins_dict if self._builtins_converter else None
        )

        self._converted.update(converted_defs)
        self._state.store_anaysis(defuse=defuse, locals=locals)

        usedef = usedef_chains(defuse)
        self._state.store_anaysis(usedef=usedef)

        # : Reachability analysis
        unreachable = get_unreachable(self._state, self._options, module_node)
        self._state.store_anaysis(unreachable=unreachable)

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
                        self._analyze_module_pass1(mod)

                    # add dependencies
                    if iteration != self._options.nested_dependencies:
                        if name != 'builtins':
                            # TODO: this could use some fine tuning
                            deps = [self._state.get_def(al).orgmodule
                                for al in (
                                    n
                                    for n in ast.walk(mod.node)
                                    if isinstance(n, ast.alias)
                                )]
                        else:
                            deps = [i.orgmodule for i in 
                                    self._builtins_converter.alias2importinfo.values()]
                        
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
        self._finalize_builtin_module()

        # after all modules have been analyzed once, we convert the builtins
        
        # : Imports analysis: complement def-use chains with import chains
        # must be done after all modules have been added
        for mod in self._state.get_all_modules():
            ChainDefUseOfImports(self._state).visit(mod.node)
        # at this point goto definition is working, for non wildcard imports names
        # : Compute __all__ and wildcard imports and fixup def-use chains for wildcard imports
        self._state._dunder_all = compute_wildcards(self._state)
