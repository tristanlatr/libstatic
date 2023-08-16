"""
Wraps interface provided by ``beniget``, and make it work with the standard `ast` library.
"""
import ast
from typing import Any, Dict, List, Mapping, Optional, Tuple, Sequence

import gast  # type:ignore
from gast.ast3 import Ast3ToGAst  # type:ignore
from beniget.beniget import ( # type:ignore
    DefUseChains as BenigetDefUseChains,
    Def as BenigetDef,
)

from .model import Cls, Func, Var, Imp, Def, Arg, Lamb, Comp, Attr, NameDef
from .imports import ParseImportedNames, ImportInfo
from .exceptions import StaticCodeUnsupported
from .shared import LocalStmtVisitor, StmtVisitor
from .arguments import iter_arguments, ArgSpec

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
                                            node.col_offset)
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
        gast2ast: Mapping[gast.AST, ast.AST],
        alias2importinfo: Mapping[ast.alias, ImportInfo],
        arg2spec: Mapping[ast.arg, ArgSpec],
    ) -> None:
        self.gast2ast = gast2ast
        self.alias2importinfo = alias2importinfo
        self.arg2spec = arg2spec
        self.converted: Dict[BenigetDef, Optional[Def]] = {}


    def convert(self, b: DefUseChains) -> Tuple[Chains, Locals, BuiltinsChains]:
        self.filename = b.filename
        chains: Chains = self._convert_chains(b.chains) # type:ignore
        builtins_chains: BuiltinsChains = self._convert_chains(b._builtins, is_builtins=True) # type:ignore
        locals = self._convert_locals(b.locals)
        return chains, locals, builtins_chains

    def _convert_definition(self, definition: BenigetDef) -> "Def|None":
        if definition in self.converted:
            return self.converted[definition]

        if not isinstance(definition.node, gast.AST):
            # a builtin
            new_definition = self._def_factory(definition.node, islive=True)
        else:
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
            # ivars will be set later
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
            d = locals_as_dict.setdefault(self.gast2ast[namespace], {})
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

class IVarsVisitor(LocalStmtVisitor):
    def __init__(self, chains:Mapping[ast.AST, Def]) -> None:
        self.chains = chains
        self.ivars: List[Attr] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args.args
        if (
            len(args) == 0
            # special methods known not to be instance methods,
            # TODO: are there more?
            or node.name in {"__new__", 
                                "__init_subclass__",
                                "__class_getitem__", 
                                "__subclasscheck__", 
                                "__subclasshook__", }
            or any((getattr(d, "id", None)
                    in {"classmethod", "staticmethod"}
                    for d in node.decorator_list))
            # we include only methods that follows the 
            # 'self' convention to exclude code like
            # class C:
            #   def stmeth(thing):
            #       thing.var = 2
            #   stmeth = staticmethod(stmeth)
            or not args[0].arg=='self'
        ):
            # not a typical instance method
            return
        self_def = self.chains[args[0]]
        for use in (u for u in self_def.users() if isinstance(u.node, ast.Name)):
            try:
                attr = next(attr for attr in use.users() 
                            if isinstance(attr, Attr) 
                            and attr.node.value is use.node)
            except StopIteration:
                continue
            else:
                self.ivars.append(attr)

    visit_AsyncFunctionDef = visit_FunctionDef

def _compute_ivars(chains: Mapping[ast.AST, Def], cls: ast.ClassDef) -> Sequence[Attr]:
    visitor = IVarsVisitor(chains)
    visitor.generic_visit(cls)
    return visitor.ivars

class ComputeInstanceVariables(StmtVisitor):
    def __init__(self, chains:Mapping[ast.AST, Def], ) -> None:
        self.chains = chains
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        cls = self.chains[node]
        assert isinstance(cls, Cls)
        cls.ivars = _compute_ivars(self.chains, node)
        self.generic_visit(node)

class ParseArgumentsInfos(StmtVisitor):        
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Any:
        self._result.update({a.node:a for a in iter_arguments(node.args)})
        self.generic_visit(node)
    def visit_Module(self, node: ast.Module) -> Mapping[ast.arg, ArgSpec]:
        self._result: Dict[ast.arg, ArgSpec] = {}
        self.generic_visit(node)
        return self._result
    visit_AsyncFunctionDef = visit_FunctionDef

def defuse_chains_and_locals(
    node: ast.Module, 
    modname: str, 
    filename: str, 
    is_package: bool,
) -> Tuple[Chains, Locals, BuiltinsChains]:
    # create gast node as well as gast -> ast mapping
    gast_node, gast2ast = ast_to_gast(node)

    # - compute beniget def-use chains
    defuse = DefUseChains(filename=filename)
    setattr(defuse, "future_annotations", True)
    setattr(defuse, "is_stub", True)
    defuse.visit(gast_node)

    # parse imports
    alias2importinfo = ParseImportedNames(modname, is_package=is_package).visit_Module(
        node
    )

    # parse function's arguments
    arg2spec = ParseArgumentsInfos().visit_Module(node)

    # convert result into standard library
    converter = BenigetConverter(gast2ast, alias2importinfo, arg2spec)
    chains, locals, builtins_defuse = converter.convert(defuse)
    # compute ivars, this mutates the Cls instances in the chains.
    ComputeInstanceVariables(chains).visit(node)
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
