"""
Technically, this is part of the analyzer.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Collection, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing import TypeGuard

from .._lib.process import TopologicalProcessor
from .._lib.model import  Def, Mod, Imp, Var
from .._lib.exceptions import StaticException
from .._lib.assignment import get_stored_value
from .._lib.shared import LocalStmtVisitor

from .state import MutableState, State

class _VisitDunderAllAssignment(ast.NodeVisitor):
    """
    Ensures that dependencies required to calculate __all__ are processed before
    going forward. Only other __all__ values will be considered as dependencies
    to calculate __all__, meaning developers should not include arbitrary names in __all__ values.

    This will not be considered::

        from mylib._impl import all_names
        __all__ = all_names + ['one', 'two']
    """

    def __init__(
        self, state: State, builder: TopologicalProcessor[str, Mod, Optional["Collection[str]"]]
    ) -> None:
        self._state = state
        self._builder = builder

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            for d in self._state.goto_defs(node, noraise=True):
                if isinstance(d.node, ast.alias):
                    imported = self._state.get_def(d.node)
                    if imported.orgname == "__all__":
                        self._builder.getProcessedObj(imported.orgmodule)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and node.attr == "__all__":
            modulename = node.value
            fullname = self._state.expand_expr(modulename)
            if fullname:
                self._builder.getProcessedObj(fullname)

class _VisitWildcardImports(LocalStmtVisitor):
    def __init__(
        self, state: State, builder: TopologicalProcessor[str, Mod, Optional["Collection[str]"]]
    ) -> None:
        self._state = state
        self._builder = builder
        self._result: Dict[ast.alias, Optional["Collection[str]"]] = OrderedDict()

    def visit_Module(
        self, node: ast.Module
    ) -> Dict[ast.alias, Optional["Collection[str]"]]:
        self.generic_visit(node)
        return self._result

    def visit_Import(self, node: Union[ast.Import, ast.ImportFrom]) -> Any:
        self.generic_visit(node)
    visit_ImportFrom = visit_Import

    def visit_alias(self, node: ast.alias) -> None:
        if node.name == "*":
            imp = self._state.get_def(node)
            mod = self._builder.getProcessedObj(imp.orgmodule)
            if mod is not None:
                try:
                    dunder_all = self._builder.result[mod]
                except KeyError:
                    self._state.msg(f"failed to resolve wildcard import", ctx=node)
                    expanded_wildcard = None
                else:
                    if dunder_all is not None:
                        expanded_wildcard = dunder_all
                    else:
                        expanded_wildcard = self._state._get_public_names(mod)

                self._result[node] = expanded_wildcard
            else:
                # not in the system
                self._state.msg(f"wildcard import from unknown module {imp.orgmodule!r}", ctx=node)
                self._result[node] = None

class _ComputeWildcards:
    def __init__(
        self, 
        state: MutableState, 
        builder: TopologicalProcessor[str, Mod, "Collection[str] | None"]
    ) -> None:
        self._state = state
        self._builder = builder

    def _process_wildcard_imports(self, node: ast.Module) -> None:

        from .driver import ChainDefUseOfImports #TODO: move this class somewhere else.
        chain_imports = ChainDefUseOfImports(self._state)

        visitor = _VisitWildcardImports(self._state, self._builder)
        alias2bnames = visitor.visit_Module(node)
        
        if alias2bnames:
            # logging
            total = len(alias2bnames)
            succesed = len([None for names in alias2bnames.values() if names is not None])
            failed = total - succesed
            self._state.msg(
                f"collected {total} wildcards "
                + ("" if not failed else f"({failed} failed) ")
                + f"from module {self._state.get_def(node).name()}",
                thresh=1,
            )
        # Create Defs for each bounded names,
        # adding new defs that replaces the wildcard,
        # this is a in-place modification to our model.

        for alias, bnames in reversed(alias2bnames.items()):
            if bnames is None:
                # no need to report failed wildcards here.
                continue
            old_def = self._state.get_def(alias)
            # for each bounded names, replaces it's usages by
            # a special Def that represent a specific name.
            for name in bnames:
                # A fictional ast node to represent a particular 
                # name wildcard imports are binding.
                new_node = ast.alias(
                    name, asname=None, lineno=alias.lineno, 
                    col_offset=getattr(alias, 'col_offset', None))
                resolved_def = Imp(new_node, islive=True, orgmodule=old_def.orgmodule, orgname=name)
                
                # TODO: We should use the modifiers for the following lines:
                self._state._locals[node].setdefault(name, []).append(resolved_def) # type: ignore
                self._state._ancestors[new_node] = self._state._ancestors[alias]

                for unbound_name in list(u for u in old_def.users() 
                                         if isinstance(u.node, ast.Name) 
                                         if isinstance(u.node.ctx, ast.Load)):
                    if unbound_name.node.id == resolved_def.name(): # type:ignore[attr-defined]
                        # cleanup (needed) over-approximations of beniget
                        for d in self._state.goto_defs(unbound_name.node):
                            self._state.remove_user(d, unbound_name)

                        # add resolved use to the chains
                        resolved_def.add_user(unbound_name)
                # add the new def to the state
                self._state.add_definition(resolved_def)
                chain_imports.visit(new_node)
            
            # should not call remove_definition(old_def) here because some names might still
            # be only bound to the wildcard in complex __all__ definitions.
    
    def _process_definitions(self, node: ast.Module) -> Optional[Def]:
        """
        Visit the defnintions of __all__ and ensure depending modules
        are processed as well.
        """
        dunder_all_def:Optional[Def] = None
        for definition in self._state.get_local(node, "__all__"):
            if isinstance(definition, Imp):
                # __all__ is an import
                self._builder.getProcessedObj(definition.orgmodule)
                dunder_all_def = definition

            elif isinstance(definition, Var):
                try:
                    # __all__ is an assignment
                    assign = self._state.get_parent_instance(
                        definition.node, (ast.Assign, ast.AnnAssign)
                    )
                    value = get_stored_value(definition.node, assign)  # type: ignore
                    if value:
                        _VisitDunderAllAssignment(self._state, self._builder).visit(
                            value
                        )
                        dunder_all_def = definition

                except StaticException:
                    # __all__ is defined but it's not an assignment or an import,
                    # we can't figure-out it's value at the moment.
                    pass

        return dunder_all_def

    def compute_wildcards(self, node: ast.Module) -> "Collection[str] | None":
        dunder_all: "Collection[str] | None" = None

        self._process_wildcard_imports(node)
        dunder_all_def = self._process_definitions(node)

        if dunder_all_def:
            known_values: "dict[str, tuple[str, ...] | list[str]]" = {
                '.'.join((mod.name(), '__all__')): _all
                for mod, _all in self._builder.result.items()
                if isinstance(_all, (tuple, list))
            }
            try:
                literal_all = self._state.literal_eval(
                    dunder_all_def.node, known_values=known_values
                )
            except StaticException as e:
                self._state.msg(
                    f"cannot evaluate value of __all__: {e}", ctx=dunder_all_def.node
                )
            else:

                def isvalid(o: Any) -> "TypeGuard[str]":
                    if isinstance(o, str) and o:
                        return True
                    # warn?
                    return False

                if isinstance(literal_all, (list, tuple)):
                    dunder_all = type(literal_all)(filter(isvalid, literal_all))  # type: ignore[assignment]

        return dunder_all


def compute_wildcards(state: MutableState) -> Mapping[Mod, "Collection[str] | None"]:
    """
    Maps ast Modules to the collection of names explicitely given in ``__all__`` variable.
    If ``__all__`` is not defined at the module scope, the result is None.

    Maps each modules to the list of names imported if one wildcard imports this module.

    A bi-product of this analysis maps each wildcard ImportFrom nodes 
    to the collection of names they are actually importing.
    """

    class WildcardsProcessor(TopologicalProcessor[str, Mod, Optional[Collection[str]]]):
        def getObj(self, name: str) -> Optional[Mod]:
            return state.get_module(name)

        def processObj(self, mod: Mod) -> Optional[Collection[str]]:
            return _ComputeWildcards(state, self).compute_wildcards(mod.node)

    return WildcardsProcessor().process(state.get_all_modules())
