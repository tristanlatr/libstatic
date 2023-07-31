"""
Wraps interface provided by C{beniget}.
"""
import ast
from typing import Dict, List, Mapping, Optional, Tuple

import gast  # type:ignore
from gast.ast3 import Ast3ToGAst  # type:ignore
from beniget.beniget import ( # type:ignore
    DefUseChains as BenigetDefUseChains,
    Def as BenigetDef,
)  

from .model import Cls, Func, Var, Imp, Def, Arg, AnonymousScope, NameDef
from .imports import ParseImportedNames, ImportInfo
from .exceptions import StaticTypeError, StaticCodeUnsupported

# beniget integration here:


class _AstToGAst(Ast3ToGAst):
    def __init__(self) -> None:
        self.mapping: Dict[gast.AST, ast.AST] = {}

    def visit(self, node: ast.AST) -> gast.AST:
        try:
            newnode = super().visit(node)
        except StaticCodeUnsupported:
            raise
        except Exception:
            raise StaticCodeUnsupported(node, 'error in ast to gast')
        
        if not isinstance(node, ast.expr_context):
            self.mapping[newnode] = node
        return newnode


def ast_to_gast(node: ast.Module) -> Tuple[gast.Module, Mapping[gast.AST, ast.AST]]:
    """
    This function returns a tuple which first element is the ``gast`` module and the second element is a
    mapping from gast nodes to standard library nodes. It should be used with caution
    since not all nodes have a corespondance. Namely, ``expr_context`` nodes and the store ``Name`` of
    ``ExceptHandler``s are not present in the mapping.
    """
    # returns a tuple: (gast node, mapping from gast node to ast node)
    _vis = _AstToGAst()
    newnode = _vis.visit(node)
    return newnode, _vis.mapping


class DefUseChains(BenigetDefUseChains):
    """
    Custom def-use builder
    """


Chains = Dict[ast.AST, Def]
Locals = Dict[ast.AST, Dict[str, List["NameDef|None"]]]


class BenigetConverter:
    def __init__(
        self,
        gast2ast: Mapping[gast.AST, ast.AST],
        alias2importinfo: Mapping[ast.alias, ImportInfo],
    ) -> None:
        self.gast2ast = gast2ast
        self.alias2importinfo = alias2importinfo
        self.converted: Dict[BenigetDef, Optional[Def]] = {}
        self.scopes: Dict[BenigetDef, Optional[BenigetDef]] = {}

    def convert(self, b: DefUseChains) -> Tuple[Chains, Locals]:
        chains = self._convert_chains(b.chains)
        locals = self._convert_locals(b.locals)
        return chains, locals

    def _convert_definition(self, definition: BenigetDef) -> "Def|None":
        if definition in self.converted:
            return self.converted[definition]

        ast_node = self.gast2ast.get(definition.node)
        if ast_node is None:
            new_definition = None
        else:
            new_definition = self._def_factory(ast_node)

        self.converted[definition] = new_definition
        if new_definition:
            for user in definition.users():
                new_user = self._convert_definition(user)
                if new_user:
                    new_definition.add_user(new_user)

        return new_definition

    def _convert_chains(self, chains: Dict[gast.AST, BenigetDef]) -> Chains:
        new_chains: Dict[ast.AST, Def] = {}
        for definition in chains.values():
            new_def = self._convert_definition(definition)
            if new_def:
                new_chains[new_def.node] = new_def
        return new_chains

    def _def_factory(self, node: ast.AST) -> "Def|None":
        # attributes are not NameDef
        if isinstance(node, ast.ClassDef):
            return Cls(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return Func(node)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            return Var(node)
        elif isinstance(node, ast.arg):
            # TODO: get kind, default value and annotation in this object.
            return Arg(node)
        elif isinstance(node, ast.alias):
            info = self.alias2importinfo.get(node)
            if info:
                return Imp(node, info.orgmodule, info.orgname)
            return None
        elif isinstance(
            node,
            (ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp),
        ):
            return AnonymousScope(node)
        elif isinstance(node, ast.Module):
            raise RuntimeError()
        else:
            return Def(node)

    def _convert_locals(self, locals: Mapping[ast.AST, List["BenigetDef"]]) -> Locals:
        locals_as_dict: Dict[ast.AST, Dict[str, List["NameDef|None"]]] = {}
        for namespace, loc_list in locals.items():
            d = locals_as_dict.setdefault(self.gast2ast[namespace], {})
            for loc in loc_list:
                converted_local = self.converted[loc]
                if __debug__:
                    assert isinstance(converted_local, (NameDef, type(None)))
                d.setdefault(loc.name(), []).append(converted_local) # type: ignore
        return locals_as_dict


def defuse_chains_and_locals(
    node: ast.Module, modname: str, filename: str, is_package: bool
) -> Tuple[Chains, Locals]:
    # create gast node as well as gast -> ast mapping
    gast_node, gast2ast = ast_to_gast(node)

    # - compute beniget def-use chains
    defuse = DefUseChains(filename=filename)
    setattr(defuse, "future_annotations", True)
    defuse.visit(gast_node)

    # parse imports
    alias2importinfo = ParseImportedNames(modname, is_package=is_package).visit_Module(
        node
    )

    # convert result into standard library
    converter = BenigetConverter(gast2ast, alias2importinfo)
    return converter.convert(defuse)


def usedef_chains(def_use_chains: Chains) -> Dict[ast.AST, List[Def]]:
    """
    Flip the Def-Use chains to generate Use-Def chains. It does not include the use of buitins.
    """
    chains: Dict[ast.AST, List[Def]] = {}
    for chain in def_use_chains.values():
        # init a empty list for all Name and alias instances
        if isinstance(chain.node, (ast.Name, ast.alias)):
            chains.setdefault(chain.node, [])

        for use in chain.users():
            chains.setdefault(use.node, []).append(chain)

    return chains
    # this does not support builtins, by design
