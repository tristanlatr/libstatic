"""
Technically, this is part of the L{analyzer}.
"""

import ast
from typing import Any, Dict, Mapping, Optional, Collection, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from .process import Processor
from .model import MutableState, Def, State, Mod, Imp, Var
from .assignment import get_stored_value
from .exceptions import StaticException

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
        self, state: State, builder: Processor[Mod, Optional["Collection[str]"]]
    ) -> None:
        self._state = state
        self._builder = builder

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            for d in self._state.goto_defs(node, noraise=True):
                if isinstance(d.node, ast.alias):
                    imported = self._state.get_def(d.node)
                    if imported.orgname == "__all__":
                        self._builder.getProcessedModule(imported.orgmodule)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and node.attr == "__all__":
            modulename = node.value
            fullname = self._state.expand_expr(modulename)
            if fullname:
                self._builder.getProcessedModule(fullname)


class _VisitWildcardImports(ast.NodeVisitor):
    def __init__(
        self, state: State, builder: Processor[Mod, Optional["Collection[str]"]]
    ) -> None:
        self._state = state
        self._builder = builder
        self._result: Dict[ast.alias, Optional["Collection[str]"]] = {}

    def visit_Module(
        self, node: ast.Module
    ) -> Dict[ast.alias, Optional["Collection[str]"]]:
        self.generic_visit(node)
        return self._result

    def visit_alias(self, node: ast.alias) -> None:
        if node.name == "*":
            imp = self._state.get_def(node)
            mod = self._builder.getProcessedModule(imp.orgmodule)
            if mod:
                dunder_all = self._builder.result.get(mod)
                if dunder_all is not None:
                    expanded_wildcard = dunder_all
                else:
                    expanded_wildcard = self._state.get_public_names(mod)
                self._result[node] = expanded_wildcard
            else:
                # not in the system
                self._result[node] = None

    def _returns(self, ob: ast.stmt) -> None:
        return

    visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _returns  # type: ignore


class _ComputeWildcards(ast.NodeVisitor):
    def __init__(
        self, state: MutableState, builder: Processor[Mod, "Collection[str] | None"]
    ) -> None:
        self._state = state
        self._builder = builder

    def _process_wildcard_imports(self, node: ast.Module) -> None:
        visitor = _VisitWildcardImports(self._state, self._builder)
        alias2bnames = visitor.visit_Module(node)
        if alias2bnames:
            self._state.msg(
                f"collected {len(alias2bnames)} wildcards "
                f"from module {self._state.get_def(node).name()}",
                thresh=1,
            )

        # Create Defs for each bounded names,
        # adding new defs that replaces the wildcard,
        # this is a in-place modification to our model.

        for alias, bnames in reversed(alias2bnames.items()):
            if not bnames:
                self._state.msg(f"wildcard could not be resolved.", ctx=alias)
                continue
            old_def = self._state.get_def(alias)
            self._state.remove_definition(old_def)
            # for each bounded names, replaces it's usages by
            # a special Def that represent a specific name.
            for name in bnames:
                # A fictional ast node to represent a particular name wildcard imports are binding.
                new_node = ast.copy_location(ast.alias(
                    name,
                    asname=None
                ), alias)
                resolved_def = Imp(new_node, orgmodule=old_def.orgmodule, orgname=name)
                self._state.add_definition(resolved_def)
                # We should use the modifiers for the following line:
                self._state._locals[node].setdefault(name, []).append(resolved_def)

                for unbound_name in list(old_def.users()):
                    if unbound_name.name() == resolved_def.name():
                        # cleanup (needed) over-approximations of beniget
                        for d in self._state.goto_defs(unbound_name.node):
                            self._state.remove_user(d, unbound_name)

                        # add new definition to the chains
                        self._state.add_user(resolved_def, unbound_name)

    def _process_definitions(self, node: ast.Module) -> Optional[Def]:
        """
        Visit the defnintions of __all__ and ensure depending modules
        are processed as well.
        """
        dunder_all_def:Optional[Def] = None
        for definition in self._state.get_local(node, "__all__"):
            if isinstance(definition, Imp):
                # __all__ is an import
                self._builder.getProcessedModule(definition.orgmodule)
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

    def visit_Module(self, node: ast.Module) -> "Collection[str] | None":
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
                    f"cannot compute value of __all__: {e}", ctx=dunder_all_def.node
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

    class WildcardsProcessor(Processor[Mod, Optional[Collection[str]]]):
        def getModule(self, name: str) -> Optional[Mod]:
            return state.get_module(name)

        def processModule(self, mod: Mod) -> Optional[Collection[str]]:
            return _ComputeWildcards(state, self).visit_Module(mod.node)

    return WildcardsProcessor().process(state.get_all_modules())
