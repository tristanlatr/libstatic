"""
Wraps interface provided by ``beniget``, and make it work with the standard `ast` library.
"""
import ast
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import gast  # type:ignore
from gast.ast3 import Ast3ToGAst  # type:ignore
from beniget.beniget import ( # type:ignore
    DefUseChains as BenigetDefUseChains,
    Def as BenigetDef,
)

from ..model import Cls, Func, Var, Imp, Def, Arg, AnonymousScope, NameDef
from .imports import ParseImportedNames, ImportInfo
from ..exceptions import StaticTypeError, StaticCodeUnsupported

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
    ``ExceptHandler`` are not present in the mapping.
    """
    # returns a tuple: (gast node, mapping from gast node to ast node)
    _vis = _AstToGAst()
    newnode = _vis.visit(node)
    return newnode, _vis.mapping

_default_builtins = BenigetDefUseChains()._builtins

class DefUseChains(BenigetDefUseChains):
    """
    Custom def-use builder with unified support for builtins.
    """

    def __init__(self, filename:Optional[str]=None, 
                 builtins:Optional[Mapping[str, Def]]=None):
        super().__init__(filename)
        if builtins:
            self._builtins = {k:d for k,d in builtins.items()}
    
    def location(self, node):
        if hasattr(node, "lineno"):
            filename = "{}:".format(
                "<unknown>" if self.filename is None else self.filename
            )
            return "{}{}:{}".format(filename,
                                            node.lineno,
                                            node.col_offset)
        else:
            return "?"
        
    def unbound_identifier(self, name, node):
        location = self.location(node)
        if name in _default_builtins:
            if name not in self._builtins:
                print(f"{location}: unbound identifier '{name}' (should be in builtins)")
                return
        print(f"{location}: unbound identifier '{name}'")

    # We really just want to map the names.

Chains = Dict[ast.AST, Def]
UseChains = Dict[ast.AST, List[Def]]
Locals = Dict[ast.AST, Dict[str, List["NameDef|None"]]]


class BenigetConverter:
    def __init__(
        self,
        gast2ast: Mapping[gast.AST, ast.AST],
        alias2importinfo: Mapping[ast.alias, ImportInfo]
    ) -> None:
        self.gast2ast = gast2ast
        self.alias2importinfo = alias2importinfo
        self.converted: Dict[BenigetDef, Optional[Def]] = {}


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
            new_definition = self._def_factory(ast_node, 
                                islive=getattr(definition, 'islive', True))

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

    def _def_factory(self, node: ast.AST, islive:bool) -> "Def|None":
        # attributes are not NameDef
        if isinstance(node, ast.ClassDef):
            return Cls(node, islive=islive)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return Func(node, islive=islive)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            return Var(node, islive=islive)
        elif isinstance(node, ast.arg):
            # TODO: get kind, default value and annotation in this object.
            return Arg(node, islive=islive)
        elif isinstance(node, ast.alias):
            info = self.alias2importinfo.get(node)
            if info:
                return Imp(node, islive=islive, 
                           orgmodule=info.orgmodule, 
                           orgname=info.orgname)
            return None
        elif isinstance(
            node,
            (ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp),
        ):
            return AnonymousScope(node, islive=islive)
        elif isinstance(node, ast.Module):
            raise RuntimeError()
        else:
            return Def(node, islive=islive)

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

class BenigetConverterBuiltins(BenigetConverter):
    # TODO: builtins module should not have special handling here, 
    # but rather we should alwasy return builint defs for every modules
    # and do the corespondance in the analyzer.
    # Plus using a subclass make it unclear
    def __init__(self, gast2ast: Mapping[gast.AST, ast.AST], 
                 alias2importinfo: Mapping[ast.alias, ImportInfo],
                 duc:DefUseChains,
                 converted: Dict[BenigetDef, Optional[Def]]) -> None:
        super().__init__(gast2ast, alias2importinfo)
        assert duc.module
        self.duc = duc
        self.module = duc.module
        self.converted = converted

        self.builtins_dict:Optional[Mapping[str, 'BenigetDef']] = {}
        builtins_dict = self.builtins_dict
        
        for d in duc.locals[duc.module]:
            name = d.name()
            # we can't trust the islive flag here, but why?
            if name in _default_builtins:
                builtins_dict[name] = d
                # If we have two (or more) definitions for this builtin name,
                # we under-approximate by taking the last defined name.
                # TODO: do better here, construct a single module state 
                # and run reachability analysis

        # link usages in the builtins module itself
        for name, d in duc._builtins.items():
            if name in builtins_dict:
                new_d = builtins_dict[name]
                for u in d.users():
                    new_d.add_user(u)

def defuse_chains_and_locals(
    node: ast.Module, 
    modname: str, 
    filename: str, 
    is_package: bool,
    builtins: Optional[Mapping[str, BenigetDef]]=None
) -> Tuple[Chains, Locals, Mapping[BenigetDef, Optional[Def]]]:
    # create gast node as well as gast -> ast mapping
    gast_node, gast2ast = ast_to_gast(node)

    # - compute beniget def-use chains
    defuse = DefUseChains(filename=filename, builtins=builtins)
    setattr(defuse, "future_annotations", True)
    setattr(defuse, "is_stub", True)
    defuse.visit(gast_node)

    # parse imports
    alias2importinfo = ParseImportedNames(modname, is_package=is_package).visit_Module(
        node
    )

    # convert result into standard library
    converter = BenigetConverter(gast2ast, alias2importinfo)
    chains, locals = converter.convert(defuse)
    return chains, locals, converter.converted


def usedef_chains(def_use_chains: Chains) -> UseChains:
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
