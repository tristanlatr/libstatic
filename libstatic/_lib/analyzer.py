import ast

from typing import Any, Union

from ..model import MutableState, Options, Mod
from .chains import defuse_chains_and_locals, usedef_chains
from .ancestors import Ancestors
from .reachability import get_unreachable
from .wildcards import compute_wildcards
from ..exceptions import StaticException, StaticCodeUnsupported
from .shared import StmtVisitor

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

    def _analyze_module_pass1(self, mod: "Mod") -> None:
        module_node = mod.node

        # - compute ancestors
        ancestors_vis = Ancestors()
        ancestors_vis.visit(module_node)
        ancestors = ancestors_vis.parents

        self._state.store_anaysis(ancestors=ancestors)

        # : Accumulate static analysis infos from beniget
        # - compute local def-use chains
        # - parsing imports
        defuse, locals = defuse_chains_and_locals(
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

    def _analyzer_pass1(self) -> None:
        processed_modules = set()
        to_process = [mod.name() for mod in self._state.get_all_modules()]
        iteration = 0

        while to_process:
            for name in list(to_process):
                if name not in processed_modules:
                    # load dependency modules if nested_dependencies is not zero
                    mod = self._state.get_module(
                        name
                    ) or self._state.add_typeshed_module(name)
                    if not mod:
                        continue

                    self._analyze_module_pass1(mod)

                    # add dependencies
                    if iteration != self._options.nested_dependencies:
                        to_process.extend(
                            self._state.get_def(al).orgmodule
                            for al in (
                                n
                                for n in ast.walk(mod.node)
                                if isinstance(n, ast.alias)
                            )
                        )

                to_process.remove(name)
                processed_modules.add(name)
            iteration += 1

    def analyze(self) -> None:
        """
        Initiate the project state.
        """
        self._analyzer_pass1()
        
        # : Imports analysis: complement def-use chains with import chains
        # must be done after all modules have been added
        for mod in self._state.get_all_modules():
            ChainDefUseOfImports(self._state).visit(mod.node)
        # at this point goto definition is working, for non wildcard imports names
        # : Compute __all__ and wildcard imports and fixup def-use chains for wildcard imports
        self._state._dunder_all = compute_wildcards(self._state)
