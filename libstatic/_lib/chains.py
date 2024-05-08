"""
Wraps interface provided by ``beniget``, and make it work with the standard `ast` library.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Mapping, Optional, Tuple, Sequence

from beniget.standard import DefUseChains as BenigetDefUseChains # type: ignore
from beniget.beniget import Def as BenigetDef # type: ignore

# TODO: This module mixes-up argument paring ans import reolving as part of the def use chains
# Even if that might have looked as a good idea, it's not and this should be separared in several analyses

from .model import Cls, Func, Var, Imp, Def, Arg, Lamb, Comp, Attr, NameDef
from .imports import ParseImportedNames, ImportInfo
from .exceptions import StaticCodeUnsupported
from .arguments import iter_arguments, ArgSpec

# beniget integration here:


class DefUseChains(BenigetDefUseChains):
    """
    Custom def-use builder.
    """

    def location(self, node):
        if hasattr(node, "lineno"):
            filename = "{}:".format(
                "<unknown>" if self.filename is None else self.filename
            )
            return "{}{}:{}".format(filename,
                                            node.lineno,
                                            getattr(node, 'col_offset', None))
        else:
            return "?"
        
    def unbound_identifier(self, name, node):
        # TODO: use reporter object
        location = self.location(node)
        print(f"{location}: unbound identifier '{name}'")

    # TODO: We really just want to map the names.

BuiltinsChains = Mapping[str, Def]
Chains = Mapping[ast.AST, Def]
UseChains = Mapping[ast.AST, Sequence[Def]]
Locals = Mapping[ast.AST, Mapping[str, Sequence["NameDef|None"]]]


class BenigetConverter:
    def __init__(
        self,
        alias2importinfo: Mapping[ast.alias, ImportInfo],
        arg2spec: Mapping[ast.arg, ArgSpec],
    ) -> None:
        self.alias2importinfo = alias2importinfo
        self.arg2spec = arg2spec
        self.converted: Dict[BenigetDef, Optional[Def]] = {}


    def convert(self, b: DefUseChains) -> Tuple[Chains, Locals, BuiltinsChains]:
        self.filename = b.filename
        chains: Chains = self._convert_chains(b.chains) # type:ignore
        builtins_chains: BuiltinsChains = self._convert_chains(b._builtins, is_builtins=True) # type:ignore
        locals = self._convert_locals(b.locals)
        return chains, locals, builtins_chains

    def _convert_definition(self, definition: BenigetDef) -> Def:
        if definition in self.converted:
            return self.converted[definition]

        if not isinstance(definition.node, ast.AST):
            # a builtin
            new_definition = self._def_factory(definition.node, islive=True)
        else:
            ast_node = definition.node
            new_definition = self._def_factory(ast_node, 
                                    islive=getattr(definition, 'islive', True))

        self.converted[definition] = new_definition
        if new_definition:
            for user in definition.users():
                new_user = self._convert_definition(user)
                new_definition.add_user(new_user)

        return new_definition

    # TODO: use overload
    def _convert_chains(self, chains: Dict[Any, BenigetDef], is_builtins:bool=False) -> 'Chains|BuiltinsChains':
        new_chains: Dict[Any, Def] = {}
        for node, definition in chains.items():
            new_def = self._convert_definition(definition)
            if new_def:
                if is_builtins:
                    # here node is actually the symbol name
                    assert isinstance(node, str)
                    new_chains[node] = new_def
                else:
                    new_chains[new_def.node] = new_def
        return new_chains

    def _def_factory(self, node: ast.AST, islive:bool) -> "Def|None":
        # attributes **are** NameDef
        if isinstance(node, ast.ClassDef):
            return Cls(node, islive=islive)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return Func(node, islive=islive)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            return Var(node, islive=islive)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            return Attr(node, islive=islive)
        elif isinstance(node, ast.arg):
            # TODO: get kind, default value and annotation in this object.
            try:
                argspec = self.arg2spec[node]
            except KeyError as e:
                raise StaticCodeUnsupported(node, 'argument was not parsed', filename=self.filename) from e
            return Arg(node, islive=islive,
                       default=argspec.default, 
                       kind=argspec.kind)
        elif isinstance(node, ast.alias):
            try:
                info = self.alias2importinfo[node]
            except KeyError as e:
                raise StaticCodeUnsupported(node, 'import was not parsed', filename=self.filename) from e
            return Imp(node, islive=islive, 
                        orgmodule=info.orgmodule, 
                        orgname=info.orgname)
        elif isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp)):
            return Comp(node, islive=islive)
        elif isinstance(node, ast.Lambda):
            return Lamb(node, islive=islive)
        elif isinstance(node, ast.Module):
            raise RuntimeError()
        else:
            # not a definition, simply a use.
            return Def(node, islive=islive)

    def _convert_locals(self, locals: Mapping[ast.AST, List["BenigetDef"]]) -> Locals:
        locals_as_dict: Dict[ast.AST, Dict[str, List["NameDef|None"]]] = {}
        for namespace, loc_list in locals.items():
            d = locals_as_dict.setdefault(namespace, {})
            for loc in loc_list:
                try:
                    converted_local = self.converted[loc]
                except KeyError:
                    # globals and non locals are not handled at the moment.
                    # Issues with the global keyword needs to be fixed first
                    # https://github.com/serge-sans-paille/beniget/issues/74
                    # https://github.com/serge-sans-paille/beniget/issues/64
                    continue
                    
                if __debug__:
                    assert isinstance(converted_local, (NameDef, type(None)))
                d.setdefault(loc.name(), []).append(converted_local) # type: ignore
        return locals_as_dict

class ParseArgumentsInfos(ast.NodeVisitor):    
    visit_Pass = visit_Break = visit_Continue = visit_Delete = visit_Global = visit_Nonlocal = visit_Import = visit_ImportFrom = lambda _,__:None # type:ignore
    
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> Any:
        self._result.update({a.node:a for a in iter_arguments(node.args)})
        self.generic_visit(node)
    def visit_Module(self, node: ast.Module) -> Mapping[ast.arg, ArgSpec]:
        self._result: Dict[ast.arg, ArgSpec] = {}
        self.generic_visit(node)
        return self._result
    visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef

def defuse_chains_and_locals(
    node: ast.Module, 
    modname: str, 
    filename: str, 
    is_package: bool,
) -> Tuple[Chains, Locals, BuiltinsChains]:
    # create gast node as well as gast -> ast mapping

    # - compute beniget def-use chains
    defuse = DefUseChains(filename=filename)
    setattr(defuse, "future_annotations", True)
    # There is somthing wrong with the stub support in beniget-ng :/ 
    # so we hard set the stub value to False here 
    # TODO: Fix this problem... This is likely due to the builtins chains and how we handle
    # these links when is_stub=False is not compatible with the builtins.pyi module :/
    setattr(defuse, "is_stub", False)
    defuse.visit(node)

    # parse imports
    alias2importinfo = ParseImportedNames(modname, is_package=is_package).visit_Module(
        node
    )

    # parse function's arguments
    arg2spec = ParseArgumentsInfos().visit_Module(node)

    # convert result into standard library
    converter = BenigetConverter(alias2importinfo, arg2spec)
    chains, locals, builtins_defuse = converter.convert(defuse)
    
    return chains, locals, builtins_defuse


def usedef_chains(def_use_chains: Chains) -> UseChains:
    """
    Flip the Def-Use chains to generate Use-Def chains. 
    """
    chains: Dict[ast.AST, List[Def]] = {}
    for chain in def_use_chains.values():
        # init a empty list for all Name and alias instances
        if isinstance(chain.node, (ast.Name, ast.alias)):
            chains.setdefault(chain.node, [])

        for use in chain.users():
            chains.setdefault(use.node, []).append(chain)

    return chains
