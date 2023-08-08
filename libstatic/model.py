"""
This module contains the def-use models, use to represent the code as well as project-wise objects.
"""

import ast
from dataclasses import dataclass
from itertools import chain
import sys
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterator,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSet,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
    Type,
    TypeVar,
    cast,
    overload,
)

from beniget.beniget import ordered_set  # type: ignore
from typeshed_client import get_stub_file, get_search_context
from typeshed_client.finder import parse_stub_file

from ._lib.shared import ast_node_name, node2dottedname
from ._lib.transform import Transform
from ._lib.asteval import LiteralValue, _LiteralEval, _GotoDefinition
from .exceptions import (
    StaticStateIncomplete,
    StaticNameError,
    StaticAttributeError,
    StaticImportError,
    StaticException,
    StaticValueError,
    StaticTypeError,
)

if TYPE_CHECKING:
    from typing import Literal, NoReturn, Protocol
else:
    Protocol = object

T = TypeVar("T", bound=ast.AST)


class _Msg(Protocol):
    def __call__(
        self, msg: str, ctx: Optional[ast.AST] = None, thresh: int = 0
    ) -> None:
        ...


class Def:
    """
    Model a use or a definition, either named or unnamed, and its users.
    """
    __slots__ = 'node', '_users', 'islive'

    def __init__(self, node:ast.AST, islive:bool=True) -> None:
        self.node = node
        self.islive = islive
        self._users: MutableSet["Def"] = ordered_set()

    def add_user(self, node: "Def") -> None:
        assert isinstance(node, Def)
        self._users.add(node)

    def name(self) -> Optional[str]:
        """
        If the node associated to this Def has a name, returns this name.
        Otherwise returns None.
        """
        return None

    def users(self) -> Collection["Def"]:
        """
        The list of ast entity that holds a reference to this node.
        """
        return self._users
    
    def __repr__(self) -> str:
        clsname = self.__class__.__qualname__
        name = self.name()
        if name:
            return f"<{clsname}(name={name})>"
        else:
            nodeclsname = self.node.__class__.__name__
            return f"<{clsname}(node=<{nodeclsname}>)>"

    def __str__(self) -> str:
        return self._str({})

    def _str(self, nodes: Dict["Def", int]) -> str:
        if self in nodes:
            return "(#{})".format(nodes[self])
        else:
            nodes[self] = len(nodes)
            return "{} -> ({})".format(
                self.name() or self.node.__class__.__name__,
                ", ".join(u._str(nodes.copy()) for u in self._users),
            )

class NameDef(Def):
    """
    Model the definition of a name.
    """

    node: Union[
        ast.Module,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Name,
        ast.arg,
        ast.alias,
    ]

    def name(self) -> str:
        assert not isinstance(self.node, ast.Module)
        return ast_node_name(self.node)

class Scope(Def):
    """
    Model a python scope.
    """

    def name(self) -> str:
        raise NotImplementedError()

class OpenScope(Scope):
    node: Union[ast.Module, ast.ClassDef]

class ClosedScope(Scope):
    """
    Closed scope have <locals>.
    """

    node: Union[
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.GeneratorExp,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
    ]


class AnonymousScope(ClosedScope):
    node: Union[ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.DictComp, ast.SetComp]

    def name(self) -> str:
        return f"<{type(self.node).__name__.lower()}>"


class Mod(NameDef, OpenScope):
    """
    Model a module definition.
    """
    __slots__ = (*Def.__slots__, '_modname', 'is_package', '_filename')

    node: ast.Module
    def __init__(self, 
                 node: ast.Module, 
                 modname: str, 
                 is_package: bool = False, 
                 filename: Optional[str] = None) -> None:
        super().__init__(node)
        self._modname = modname
        self.is_package = is_package
        self._filename = filename

    def name(self) -> str:
        return self._modname

    def filename(self) -> str:
        return self._filename or self._modname


class Cls(NameDef, OpenScope):
    """
    Model a class definition.
    """

    node: ast.ClassDef


class Func(NameDef, ClosedScope):
    """
    Model a function definition.
    """

    node: Union[ast.FunctionDef, ast.AsyncFunctionDef]


class Var(NameDef):
    """
    Model a variable definition.
    """

    node: ast.Name


class Arg(NameDef):
    """
    Model a function argument definition.
    """

    node: ast.arg


class Imp(NameDef):
    """
    Model an imported name definition.
    """
    __slots__ = (*Def.__slots__, 'orgmodule', 'orgname')
    
    node: ast.alias
    def __init__(self, 
                 node: ast.alias, 
                 islive: bool, 
                 orgmodule: str, 
                 orgname: Optional[str] = None) -> None:
        super().__init__(node, islive)
        self.orgmodule = orgmodule
        self.orgname = orgname

    def target(self) -> str:
        """
        Returns the qualified name of the the imported symbol.
        """
        if self.orgname:
            return f"{self.orgmodule}.{self.orgname}"
        else:
            return self.orgmodule


### Project-wide state and accessors

class _MinimalState(Protocol):
    msg: _Msg
    def get_filename(self, n:ast.AST)->'str|None':...

    def get_local(self, node:ast.AST, name:str) -> List[Optional[Def]]:... # can be empty
    # these returned lists should never be empty, raise exception instead.
    def get_attribute(self, node:ast.AST, name:str) -> List['Def']:...
    @overload
    def get_def(self, node: "ast.alias") -> Imp:
        ...
    @overload
    def get_def(self, node:ast.AST) -> 'Def':...
    def goto_defs(self, node:ast.AST) -> List['Def']:...
    def get_parent_instance(
        self, node: ast.AST, cls: "Type[T]|Tuple[Type[T],...]"
    ) -> T:...
    def expand_expr(self, node:ast.AST) -> 'str|None':
        ...

class State(_MinimalState):
    """
    The `Project`'s state.
    """

    def __init__(self, msg: _Msg) -> None:
        self.msg = msg

        self._modules: Dict[str, "Mod"] = {}
        """
        Mapping from module names to Mod instances.
        """

        self._unreachable: Set[ast.AST] = set()
        """
        Set of unreachable nodes.
        """

        self._locals: Dict[ast.AST, Dict[str, List[Optional[NameDef]]]] = {}
        """
        Mapping of locals.
        """

        self._ancestors: Dict[ast.AST, List[ast.AST]] = {}
        """
        Mapping of AST nodes to the list of their parents.
        """

        self._def_use_chains: Dict[ast.AST, Def] = {}
        """
        Def-Use chains.
        """

        self._use_def_chains: Dict[ast.AST, List[Def]] = {}
        """
        Use-Def chains.
        """

        self._dunder_all: Mapping["Mod", "Collection[str]|None"] = {}
        """
        Mapping from Mod instances explicit ``__all__`` values or None.
        """
    
    def _raise_node_not_in_chains(self, e: KeyError, node:ast.AST) -> 'NoReturn':
        try:
            self.get_parents(node)
        except StaticException:
            # node is not in the system
            raise StaticStateIncomplete(node, "node not in the system") from e
        else:
            # node is not a supposed to be a use,
            # like ast.Return/Delete/While/For and many others, to get the full list of
            # beniget visit_ methods that do not instanciate a new Def use:
            # pyastgrep './/FunctionDef[contains(@name, "visit_")][not(contains(body//Call/func/Name/@id, "Def"))]'
            raise StaticValueError(node, "node is not a use or a definition") from e

    @overload # type: ignore[override]
    def get_def(self, node: "ast.alias") -> Imp:
        ...

    @overload
    def get_def(self, node: "ast.Module") -> Mod:
        ...

    @overload
    def get_def(self, node: "ast.FunctionDef") -> Func:
        ...

    @overload
    def get_def(self, node: "ast.AsyncFunctionDef") -> Func:
        ...

    @overload
    def get_def(self, node: "ast.ClassDef") -> Cls:
        ...

    @overload
    def get_def(self, node: "ast.AST", noraise: "Literal[False]" = False) -> Def:
        ...

    @overload
    def get_def(self, node: "ast.AST", noraise: "Literal[True]") -> Optional[Def]:
        ...

    def get_def(
        self, node: "ast.AST", noraise: bool = False
    ) -> Optional[Union["Def", "Mod"]]:
        """
        Def-Use chains accessor: returns the L{Def} instance  for this node.

        :raises StaticValueError: If the node is not a registered use or definition.
        """
        try:
            return self._def_use_chains[node]
        except KeyError as e:
            if noraise:
                return None
            self._raise_node_not_in_chains(e, node)

    # @overload
    # def goto_def(self, node: "ast.alias", noraise: "Literal[False]" = False) -> NameDef:
    #     ...

    # @overload
    # def goto_def(
    #     self, node: "ast.alias", noraise: "Literal[True]"
    # ) -> Optional[NameDef]:
    #     ...

    # @overload
    # def goto_def(self, node: "ast.Name", noraise: "Literal[False]" = False) -> NameDef:
    #     ...

    # @overload
    # def goto_def(self, node: "ast.Name", noraise: "Literal[True]") -> Optional[NameDef]:
    #     ...

    @overload
    def goto_def(self, node: "ast.AST", noraise: "Literal[False]" = False) -> Def:
        ...

    @overload
    def goto_def(self, node: "ast.AST", noraise: "Literal[True]") -> Optional[Def]:
        ...

    def goto_def(self, node: ast.AST, noraise: bool = False) -> Optional[Def]:
        """
        Use-Def chains accessor that returns only one L{Def}, or raise `StaticException`.
        It returns the last def in the list. It does not ensure that the list is only
        composed by one element.
        
        :see: `goto_defs`
        """
        try:
            return self.goto_defs(node)[-1]
        except StaticException:
            if noraise:
                return None
            raise

    def goto_defs(self, node: ast.AST, noraise: bool = False) -> List["Def"]:
        """
        Use-Def chains accessor: returns the definition points of this use.

        :note: It does not recurse on follow-up definitions in case of aliases
            and it does not work for builtins at the moment.

        :raises StaticException: If the node is unbound or unknown.
        """
        try:
            defs = self._use_def_chains[node]
            if isinstance(node, ast.Name):
                # make sure only namedefs with same name are in the list.
                # this is a band-aid fix for https://github.com/serge-sans-paille/beniget/issues/63
                # see test_chains.py::..::test_annassign
                def f(d:Def) -> bool:
                    return d.name()==node.id
                defs = list(filter(f, defs))
            if len(defs) == 0:
                if isinstance(node, ast.alias):
                    raise StaticImportError(node, filename=self.get_filename(node))
                raise StaticNameError(node, filename=self.get_filename(node))            
            return defs
        except StaticException as e:
            if noraise:
                return []
            raise
        except KeyError as e:
            if noraise:
                return []
            self._raise_node_not_in_chains(e, node)

    def get_module(self, name: str) -> Optional["Mod"]:
        """
        Returns the module with the given name if it's in 
        the system, else None.
        """
        return self._modules.get(name)

    def get_all_modules(self) -> Iterable["Mod"]:
        """
        Iterate over all modules in the project.
        """
        return self._modules.values()

    def get_sub_module(
        self,
        mod: Union["Mod", ast.Module],
        name: str,
    ) -> Optional["Mod"]:
        """
        Get a sub-module of the given module.
        """
        if isinstance(mod, ast.AST):
            mod = self.get_def(mod)
        return self._modules.get(f"{mod.name()}.{name}")

    def get_parent_module(
        self,
        mod: Union["Mod", ast.Module],
    ) -> Optional["Mod"]:
        """
        Get the parent package of the given module.

        Returns None if the given module is a root 
        module or the parent package is not found in the system.
        """
        if isinstance(mod, ast.AST):
            mod = self.get_def(mod)
        if '.' not in mod.name():
            return None
        return self._modules.get('.'.join(mod.name().split('.')[:-1]))

    def get_locals(
        self, node: Union["Mod", "Def", ast.AST]
    ) -> Mapping[str, List[Optional["NameDef"]]]:
        """
        Get the mapping of locals of the given ``node``.
        """
        if isinstance(node, Def):
            node = node.node
        try:
            return self._locals[node]
        except KeyError as e:
            raise StaticValueError(node, "node has no locals") from e

    def get_local( #type:ignore[override]
        self, node: Union["Mod", "Def", ast.AST], name: str
    ) -> List[Optional["NameDef"]]: 
        """
        Get the definitions of the given ``name`` in scope ``node``.
        """
        try:
            return self.get_locals(node)[name]
        except KeyError:
            return []

    def get_attribute( # type: ignore[override]
        self,
        node: Union[ast.ClassDef, ast.Module, Mod, Cls],
        name: str,
        *,
        ignore_locals: bool = False,
        noraise: bool = False
    ) -> List[NameDef]:
        """
        Get ``scope`` attributes definitions matching the ``name``.
        It calls both `get_local()` and `get_sub_module()`.

        :raises StaticAttributeError: If the attribute is not found.
        """
        # TODO: Handle {"__name__", "__doc__", "__file__", "__path__", "__package__"}?
        # TODO: Handle {__class__, __module__, __qualname__}?
        # TODO: Handle instance variables?
        # TODO: Handle looking up in super classes?

        if isinstance(node, ast.AST):
            node = self.get_def(node) # type: ignore
        if not isinstance(node, (Mod, Cls)):
            if noraise:
                return []
            raise StaticTypeError(node, expected='Module or Class')
        if not ignore_locals:
            values = [v for v in self.get_local(node, name) if v]
        else:
            values = []
        if not values and isinstance(node, Mod) and node.is_package:
            # a sub-package
            sub = self.get_sub_module(node, name)
            if sub:
                return [sub]
        if values:
            return values
        if noraise:
            return []
        raise StaticAttributeError(node, attr=name)

    def get_dunder_all(self, mod: "Mod") -> "Collection[str]|None":
        """
        Get the computed value for the ``__all__`` variable of this module.

        If ``__all__`` variable is not defined or too complex returns None.

        :raises StaticStateIncomplete: If no information is registered for the module ``mod``.
        """
        try:
            return self._dunder_all[mod]
        except KeyError as e:
            raise StaticStateIncomplete(mod, "not information in the system") from e

    def _get_public_names(self, mod: Union["Mod", ast.Module]) -> Collection[str]:
        """
        In the absence of definition of __all__, we use this function to
        compute names bound when wildcard importing the given module.
        """
        return list(
            dict.fromkeys(
                (
                    n
                    for n in self.get_locals(mod).keys()
                    if not n.startswith("_") and n != "*"
                )
            )
        )

    def get_all_names(self, mod: Union["Mod", ast.Module]) -> Collection[str]:
        """
        Returns all names bound when wildcard importing the given module.

        :note: If ``__all__`` is defined, it simply returns the computed literal value.
            No checks is done to verify if names are actually defined.
        :raises StaticException: If something went wrong.
        """
        __all__ = self.get_dunder_all(
            mod if isinstance(mod, Mod) else self.get_def(mod)
        )
        if __all__ is not None:
            return __all__
        return self._get_public_names(mod)

    def get_parent(self, node: ast.AST) -> ast.AST:
        """
        Returns the direct parent of the given node in the syntax tree.

        :raises StaticValueError: If node is a module, is has no parents.
        """
        try:
            return self.get_parents(node)[-1]
        except IndexError as e:
            if isinstance(node, ast.Module):
                raise StaticValueError(
                    node, f"a module does not have parents in the syntax tree"
                ) from e
            raise StaticStateIncomplete(node, "missing parent") from e

    def get_parents(self, node: ast.AST) -> List[ast.AST]:
        """
        Returns all syntax tree parents of the node in the syntax tree up to the root module.

        :raises StaticStateIncomplete: If no parents informations is available.
        """
        try:
            return self._ancestors[node]
        except KeyError as e:
            raise StaticStateIncomplete(node, "no parents in the system") from e

    def get_parent_instance(
        self, node: ast.AST, cls: "Type[T]|Tuple[Type[T],...]"
    ) -> T:
        """
        Returns the first parent of the node in the syntax tree matching the given type info.

        :raises StaticValueError: If the the node has no parents of the requested type.
        """
        # special case module access for speed.
        if isinstance(cls, type) and issubclass(cls, ast.Module):
            try:
                mod = next(iter(self.get_parents(node)))
            except StopIteration:
                pass
            else:
                if isinstance(mod, cls):
                    return mod  # type: ignore
        for n in reversed(self.get_parents(node)):
            # TODO: Use TypeGard annotation
            if isinstance(n, cls):
                return n  # type: ignore
        raise StaticValueError(node, f"node has no parent of type {cls}")

    def get_root(self, node: Union[ast.AST, Def]) -> Mod:
        """
        If this node is a module, returns it's `Mod` instance,
        else find the parent Module and return it's `Mod` instance.

        :raises StaticException: If something is wrong.
        """
        if isinstance(node, Def):
            node = node.node
        if isinstance(node, ast.Module):
            return self.get_def(node)
        return self.get_def(
            self.get_parent_instance(node, ast.Module))

    def get_filename(self, node: ast.AST) -> Optional[str]:
        """
        Returns the filename of the given ast node.
        If the node does not exist in the system, it returns None.
        """
        try:
            return self.get_root(node).filename()
        except StaticException:
            return None

    def is_reachable(self, node: ast.AST) -> bool:
        """
        Whether the node is reachable.
        """
        return node not in self._unreachable

    # def dump(self) -> 'list[dict[str, Any]]':
    #     def _dump_mod(_m:Mod) -> 'dict[str, Any]':
    #         return {
    #             'is_package':_m.is_package,
    #             'modname':_m.name(),
    #             'node':ast2json(_m.node)
    #         }
    #     return [_dump_mod(m) for m in self._modules.values()]
    @overload
    def get_enclosing_scope(self, definition: Mod) -> None: # type:ignore[misc]
        ...
    @overload
    def get_enclosing_scope(self, definition: Def) -> "Scope":
        ...
    @overload
    def get_enclosing_scope(self, definition: ast.AST) -> "Scope":
        ...
    @overload
    def get_enclosing_scope(self, definition: ast.Module) -> None: # type:ignore[misc]
        ...
    def get_enclosing_scope(self, definition: Union[Def, ast.AST]) -> "Scope|None":
        """
        Get the first enclosing scope of this use or deinition.
        Returns None only of the definition is a Module.
        """
        if isinstance(definition, Def):
            definition = definition.node
        if isinstance(definition, ast.Module):
            return None
        enclosing_scope = self.get_parent_instance(
            definition,
            (
                ast.SetComp,
                ast.DictComp,
                ast.ListComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Module,
            ),
        )
        s = self.get_def(enclosing_scope)
        assert isinstance(s, Scope)
        return s # type: ignore

    def get_all_enclosing_scopes(self, definition: Def) -> Sequence[Scope]:
        """
        Returns all scopes enclosing this definition.
        """
        
        parent = self.get_enclosing_scope(definition)
        scopes = [parent]
        while parent:
            scopes.append(parent)
            parent = self.get_enclosing_scope(parent)
        return scopes # type:ignore

    def get_qualname(self, definition: Union[NameDef, Scope]) -> "str":
        """
        Returns the qualified name of this definition.
        If the definition is an imported name, returns the qualified 
        name of the the imported symbol.

        A qualified named a name by wich a definition can be found.
        The same object could have several qualified
        named depending on where it's imported.
        """
        if isinstance(definition, Imp):
            return definition.target()
        name = definition.name()
        scope = self.get_enclosing_scope(definition)
        if isinstance(scope, ClosedScope):
            name = f"<locals>.{name}"
        elif scope is None:
            # modules
            return name
        return f"{self.get_qualname(scope)}.{name}"

    def goto_symbol_def(self, scope: Scope, name:str, *, is_annotation:bool=False) -> List[NameDef]:
        """
        Simple, lazy identifier -> defs resolving.

        "Lookup" a name in the context of the provided scope, it does not use the chains
        Note that nonlocal and global keywords are ignored by this function.

        >>> p = Project()
        >>> m = p.add_module(ast.parse('from twisted.web.template import Tag as T;'), 'test')
        >>> p.analyze_project()
        >>> p.state.goto_symbol_def(m, 'T')
        [<Imp(name=T)>]

        :raise StaticNameError: For builtin or unbound names.
        """
        def _get_lookup_scopes() -> List[Scope]:
            # heads[-1] is the direct enclosing scope and heads[0] is the module.
            # returns a list based on the elements of heads, but with
            # the ignorable scopes removed. Ignorable in the sens that the lookup
            # will never happend in this scope for the given context.

            heads = list((*self.get_all_enclosing_scopes(scope), scope))
            try:
                direct_scope = heads.pop(-1) # this scope is the only one that can be a class
            except IndexError:
                raise StaticStateIncomplete(scope.node, 'missing parent scope')
            try:
                global_scope = heads.pop(0)
            except IndexError:
                # we got only a global scope
                return [direct_scope]
            # more of less modeling what's described here.
            # https://github.com/gvanrossum/gvanrossum.github.io/blob/main/formal/scopesblog.md
            other_scopes = [s for s in heads if isinstance(s, (ClosedScope))]
            return [global_scope] + other_scopes + [direct_scope]
        
        def _lookup() -> List[NameDef]:
            context = scopes.pop()
            defs:List[NameDef] = []
            for loc in self.get_local(context, name):
                if loc and loc.islive:
                    defs.append(loc)
            if defs:
                return defs
            elif len(scopes)==0:
                raise StaticNameError(name, filename=self.get_filename(scope.node))
            return _lookup()

        scopes = _get_lookup_scopes()
        scopes_len = len(scopes)
        
        if scopes_len>1 and is_annotation:
            # start by looking at module scope first,
            # then try the theoretical runtime scopes.
            # putting the global scope last in the list so annotation are
            # resolve using he global namespace first. this is the way pyright does.
            scopes.append(scopes.pop(0))
        
        return _lookup()

    def expand_name(self, scope: Scope, name: str, is_annotation:bool=False) -> "str|None":
        """
        Resove a dottedname to it's qualified name.
        by using information only available in the current module.

        Only the first name in the dottedname is resolved,
        does not recurse in attributes definitions,
        simply append the rest of the names at the end.

        >>> p = Project()
        >>> m = p.add_module(ast.parse('from twisted.web.template import Tag as T'), 'test')
        >>> p.analyze_project()
        >>> p.state.expand_name(m, 'T.something') # expand 'T' in the context of m
        'twisted.web.template.Tag.something'

        Returns None is the name is unbound.
        """
        assert name
        dottedname = name.split('.')
        try:
            top_level_definition = self.goto_symbol_def(scope, dottedname[0], is_annotation=is_annotation)[-1]
        except StaticException:
            return None
        return ".".join((self.get_qualname(top_level_definition), *dottedname[1:]))

    def expand_expr(self, node: ast.AST) -> "str|None":
        """
        Resove a name expression to it's qualified name.
        by using information only available in the current module.

        Only the first name in the expression is resolved,
        does not recurse in attributes definitions,
        simply append the rest of the names at the end.

        >>> p = Project()
        >>> node = ast.parse('from twisted.web.template import Tag as T; T')
        >>> p.add_module(node, 'test')
        <Mod(name=test)>
        >>> p.analyze_project()
        >>> use = node.body[-1].value
        >>> p.state.expand_expr(use)
        'twisted.web.template.Tag'

        Returns None if the name is unbound or the expreesion is not composed by names.
        """

        dottedname = node2dottedname(node)
        if not dottedname:
            # not composed by names
            return None
        # If node2dottedname() returns something, the expression is composed by one Name
        # and potentially multiple Attribute instance. So the following line is safe.
        top_level_name = next(
            name for name in ast.walk(node) if isinstance(name, ast.Name)
        )
        top_level_definition = self.goto_def(top_level_name, noraise=True)
        if not isinstance(top_level_definition, NameDef):
            # unbound name
            return None
        return ".".join((self.get_qualname(top_level_definition), *dottedname[1:]))

    def literal_eval(
        self, 
        node: ast.AST,
        *,
        known_values: Optional[Mapping[str, Any]] = None,
        raise_on_ambiguity: bool = False,
        follow_imports: bool = False,
    ) -> LiteralValue:
        """
        Powerfull ``ast.literal_eval()`` function. Does not support dicts at the moment.

        >>> p = Project()
        >>> node = ast.parse('from x import x;e="bar";x+["1", 2+3, e]')
        >>> node2 = ast.parse('from test import e;"best "+e')
        >>> p.add_module(node, 'test')
        <Mod(name=test)>
        >>> p.add_module(node2, 'test2')
        <Mod(name=test2)>
        >>> p.analyze_project()
        >>> expr = node.body[-1].value
        >>> p.state.literal_eval(expr, known_values={'x':['foo']})
        ['foo', '1', 5, 'bar']
        >>> expr2 = node2.body[-1].value
        >>> p.state.literal_eval(expr2)
        Traceback (most recent call last):
        libstatic.exceptions.StaticUnknownValue: test2:1:17: Unkown value: test.e
        >>> p.state.literal_eval(expr2, follow_imports=True)
        'best bar'
        
        """
        visitor = _LiteralEval(
            self,
            known_values=known_values or {},
            raise_on_ambiguity=raise_on_ambiguity,
            follow_imports=follow_imports,
        )
        result = visitor.visit(node, [])
        if isinstance(result, ast.AST):
            raise StaticTypeError(result, expected="literal")
        return result

    def goto_definition(
        self,
        node: ast.AST,
        *,
        raise_on_ambiguity: bool = False,
        follow_aliases: bool = False,
        follow_imports: bool = True,
    ) -> 'Def':
        r"""
        Go to the genuine definition of this expression.
        This is not a simple use-def chains accessor, it's recursive.
        By default it follows imports but not aliases.

        >>> p = Project()
        >>> _ = p.add_module(ast.parse('def deprecated(f):...'), 'deprecated')
        >>> src1 = p.add_module(ast.parse('''\
        ... from deprecated import deprecated
        ... @deprecated
        ... def f():...'''), 'src1')
        >>> src2 = p.add_module(ast.parse('''\
        ... import src1
        ... @src1.deprecated
        ... class C:...'''), 'src2')
        >>> p.analyze_project()
        >>> func_dec = src1.node.body[-1].decorator_list[0]
        >>> p.state.goto_definition(func_dec)
        <Func(name=deprecated)>
        >>> cls_dec = src1.node.body[-1].decorator_list[0]
        >>> p.state.goto_definition(cls_dec)
        <Func(name=deprecated)>
        """
        visitor = _GotoDefinition(
            self,
            raise_on_ambiguity=raise_on_ambiguity,
            follow_aliases=follow_aliases,
            follow_imports=follow_imports,
        )

        definition = visitor.visit(node, [])
        return self.get_def(definition)

    def _goto_references(self, definition:NameDef, seen:Set[Def]) -> Iterator[Def]:
        """
        Finds all Name or Import references, it follows imports, but bot aliases.
        """
        def _refs(dnode:Def) -> Iterable[Def]:
            if dnode in seen:
                return
            seen.add(dnode)
            if isinstance(dnode, Imp):
                # follow imports
                yield dnode
                for u in dnode.users():
                    yield from _refs(u)
                # TODO: follow aliases
            elif isinstance(dnode.node, ast.Name) and dnode.node.ctx.__class__.__name__=='Load':
                yield dnode
        for u in definition.users():
            yield from _refs(u)
    
    def _goto_attr_references(self, definition:NameDef, seen:Set[Def]) -> Iterator[Def]:
        """
        Find attribute refernces. 

        Pitfalls:
         - inherited attributes access in subclasses are not considered
        """
        seen.add(definition)
        
        # determine if this definition can be the target of a resolvable attribute access.
        # definition might be inside a class that it killed
        if not definition.islive or any(isinstance(e, 
                (ClosedScope)) or not getattr(e, 'islive', True) for e 
               in self.get_all_enclosing_scopes(definition)):
            return

        if isinstance(definition, Mod):
            module = self.get_parent_module(definition)
            if not module:
                return
            parent_qualname = module.name()
            def_name = definition.name().split('.')[-1]
        else:
            module = self.get_root(definition)
            if not module:
                return
            parent_qualname = self.get_qualname(self.get_enclosing_scope(definition))
            def_name = definition.name()

        # get_qualname returns the target name for imports, so we make sure 
        # we have the right name here by calling get_qualname on the enclosing scope only
        diffnames = [*parent_qualname.split('.'), def_name][len(module.name().split('.')):]
        
        # make sure it supports access accross sub-modules, even if there are chances that
        # it raises a runtime error if the module has not been imported, it still counts as references.
        while module:
            seen.add(module)
            for ref in self._goto_references(module, seen.copy()):
                refs = [ref]
                if isinstance(ref, Imp):
                    if ref not in seen:
                        refs = list(self._goto_attr_references(ref, seen))
                for ref in refs:                    
                    attr_node = parent_node = self.get_parent(ref.node)
                    index = 0
                    # check whether the node is part of an attribute and 
                    # compare it's dotted name with the diffnames
                    while isinstance(parent_node, ast.Attribute) and index < len(diffnames):
                        if parent_node.attr != diffnames[index]:
                            break
                        attr_node = parent_node
                        parent_node = self.get_parent(parent_node)
                        index += 1
                    else:
                        if index:
                            # It has not break and did a loop, meaning we found a match
                            attr_def = self.get_def(attr_node)
                            if attr_def not in seen:
                                seen.add(attr_def)
                                yield attr_def
            
            module = self.get_parent_module(module)
            if module:
                diffnames = [module.name().split('.')[-1], *diffnames]
        
    def goto_references(self, definition:Union[NameDef, ast.AST]) -> Iterator[Def]:
        r"""
        Finds all ``Name`` and ``Attribute`` references pointing to the given definition.

        >>> p = Project()
        >>> dep = p.add_module(ast.parse('def deprecated(f):...'), 'deprecated')
        >>> _ = p.add_module(ast.parse('''\
        ... from deprecated import deprecated
        ... @deprecated
        ... def f():...'''), 'src1')
        >>> _ = p.add_module(ast.parse('''\
        ... import src1
        ... @src1.deprecated
        ... class C:...'''), 'src2')
        >>> p.analyze_project()
        >>> dep_func = dep.node.body[-1]
        >>> list(p.state.goto_references(dep_func))
        [<Def(node=<Name>)>, <Def(node=<Attribute>)>]

        """
        if isinstance(definition, ast.AST):
            definition = self.get_def(definition) # type:ignore
        if not isinstance(definition, NameDef):
            raise StaticTypeError(definition, expected='NameDef')
        name_references = self._goto_references(definition, set())
        imports_references = []
        for ref in name_references:
            if isinstance(ref, Imp):
                imports_references.append(ref)
            else:
                yield ref
        yield from self._goto_attr_references(definition, set())
        # handle import aliases?
        for imp in imports_references:
            yield from self._goto_attr_references(imp, set())

    def get_defs_from_qualname(self, qualname:str) -> List[Def]:
        r"""
        Finds the definitions having the given qualname.

        >>> p = Project()
        >>> node = ast.parse('class Reactor:\n class System:\n  target = "win32"')
        >>> p.add_module(node, 'test')
        <Mod(name=test)>
        >>> p.analyze_project()
        >>> p.state.get_defs_from_qualname('test.Reactor.System.target')
        [<Var(name=target)>]
        """
        def find(current:List[Def], path:Sequence[str]) -> List[Def]:
            curr, *parts = path
            while curr:
                current = list(chain.from_iterable(
                    self.get_attribute(o, curr, noraise=noraise) for o in current)) # type: ignore
                if not current or not parts:
                    break
                curr, *parts = parts
            return current

        # support getting modules by name in modules mapping
        noraise=True
        names:Tuple[str, ...] = ()
        module: Optional[Def] = None
        qnameparts = qualname.split('.')
        while qnameparts:
            module = self.get_module('.'.join(qnameparts))
            if module:
                break
            *qnameparts, name = qnameparts
            names = (name, ) + names
        if module:
            if not names:
                return [module]
            ob = find([module], names)
            if ob:
                return ob
        if module:
            noraise=False
            # should raise
            return find([module], names)
        else:
            raise StaticStateIncomplete(qualname, f'{qualname!r} not in the system')

class SingleModuleState(State):
    """
    A state subclass that is internally used to run the reachability analysis.
    """
    
    def __init__(self, node:ast.Module, modname:str, filename:str, 
                 defchains: Mapping[ast.AST, Def], 
                 usechains: Mapping[ast.AST, Sequence[Def]], 
                 locals: Mapping[ast.AST, Mapping[str, Sequence[Def]]], 
                 ancestors: Mapping[ast.AST, List[ast.AST]]):
        
        mod = Mod(node, modname, filename=filename)
        self._modules[modname] = mod
        self._def_use_chains.update(defchains)
        self._use_def_chains.update(usechains)
        self._locals.update(locals)
        self._ancestors.update(ancestors)

def _load_typeshed_mod_spec(modname: str, search_context:Any) -> Tuple[Path, ast.Module, bool]:
    path = get_stub_file(modname, search_context=search_context)
    if not path:
        raise ModuleNotFoundError(name=modname)
    is_package = path.stem == "__init__"
    return path, cast(ast.Module, parse_stub_file(path)), is_package


class MutableState(State):
    """
    Class providing modifiers for the `State`.

    Among others, it that ensures that modifications of the def-use chains
    are replicated in the use-def chains as well, as if they form a unique
    data structure.
    """
    search_context = None


    def add_typeshed_module(self, modname: str) -> "Mod|None":
        """
        Add a module from typeshed or from locally installed .pyi files or typed packages.
        """
        search_context = self.search_context
        if search_context is None:
            # cache search_context
            search_context = self.search_context = get_search_context()
        try:
            path, modast, is_pack = _load_typeshed_mod_spec(modname, search_context)
        except ModuleNotFoundError as e:
            self.msg(f'stubs not found for module {modname!r}')
            return None
        except Exception as e:
            self.msg(f'error loading stubs for module {modname!r}, {e.__class__.__name__}: {e}')
            return None
        return self.add_module(modast, modname, is_package=is_pack, filename=path.as_posix())

    def add_module(
        self, node: ast.Module, name: str, *, is_package: bool, filename: Optional[str]=None
    ) -> "Mod":
        """
        Adds a module to the project.
        All modules should be added before calling `analyze_project()`.
        This will slightly transform the AST... see `Transform`.

        :raises StaticValueError: If the module name is already in the project.
        """
        if name in self._modules:
            raise StaticValueError(node, f"duplicate module {name!r}")
        # TODO: find an extensible way to transform the tree
        Transform().transform(node)
        mod = Mod(node, name, is_package=is_package, filename=filename)
        self._modules[name] = mod
        # add module to the chains
        self._def_use_chains[node] = mod
        return mod

    # use-def-use structure

    def _add_usedef(self, use: "Def", definition: "Def") -> None:
        self._use_def_chains.setdefault(use.node, []).append(definition)

    def add_definition(self, definition: "Def") -> None:
        assert definition.node not in self._def_use_chains
        self._def_use_chains[definition.node] = definition
        for u in definition.users():
            self._add_usedef(u, definition)

    def add_user(self, definition: "Def", use: "Def") -> None:
        definition.add_user(use)
        self._add_usedef(use, definition)

    def remove_user(self, definition: "Def", use: "Def") -> None:
        # TODO: use discard when the beniget version > 0.4.1
        definition._users.values.pop(use) # type: ignore
        self._use_def_chains[use.node].remove(definition)

    def remove_definition(self, definition: "Def") -> None:
        del self._def_use_chains[definition.node]
        # avoiding RuntimeError: OrderedDict mutated during iteration.
        for use in tuple(definition.users()):
            self.remove_user(definition, use)

    # first pass updates

    def store_anaysis(
        self,
        *,
        defuse: "Dict[ast.AST, Def]|None" = None,
        locals: "Dict[ast.AST, Dict[str, List[NameDef|None]]]|None" = None,
        ancestors: "Dict[ast.AST, List[ast.AST]]|None" = None,
        usedef: "Dict[ast.AST, List[Def]]|None" = None,
        unreachable: "Set[ast.AST]|None" = None,
    ) -> None:
        self._def_use_chains.update(defuse) if defuse else ...
        self._locals.update(locals) if locals else ...
        self._ancestors.update(ancestors) if ancestors else ...
        self._use_def_chains.update(usedef) if usedef else ...
        self._unreachable.update(unreachable) if unreachable else ...
    
    # custom created nodes, TODO: this might be a bad idea.

    def adopt_expression(self, expr:ast.expr, ctx:Scope) -> None:
        """
        Updates the ancestors and the chains
        such that this manually constructed expression 
        is usable in the project like other nodes.
        This can be useful for instance to register unstringed
        annotations into the state.

        Warlus operator inside adopted expression is 
        not supported: the expression should be pure.
        """
        # PartialDefUseChains = object
        # module_node = ast.Module(body=[ast.Expr(expr)])
        # defuse = PartialDefUseChains(module_node,
        #                              state=self, ctx=ctx)

    # loading

    # def load(self, data:'list[dict[str, Any]]') -> None:
    #     for mod_spec in data:
    #         assert all(k in mod_spec for k in ['node', 'modname', 'is_package'])
    #         self.add_module(json2ast(mod_spec['node']),
    #                        mod_spec['modname'],
    #                        is_package=mod_spec['is_package'])


@dataclass(frozen=True)
class Options:
    python_version: Optional[Tuple[int, int]] = None
    platform: Optional[str] = None
    nested_dependencies: int = 0
    outstream: TextIO = sys.stdout
    verbosity: int = 0


class Project:
    """
    A project is a high-level class to analyze a collection of modules together.
    """

    def __init__(self, **kw: Any) -> None:
        """
        Create a new project.

        :param kw: All parameters are passed to `Options` constructor.
        """
        self.options = Options(**kw)
        self.state: State = MutableState(msg=self.msg)

    def analyze_project(self) -> None:
        """
        Put the project's in it's final, analyzed state.

        .. note: This should only be called once.
        """
        t0 = time.time()

        from ._lib.analyzer import Analyzer

        Analyzer(cast(MutableState, self.state), self.options).analyze()

        t1 = time.time()
        self.msg(f"analysis took {t1-t0} seconds", thresh=1)

    def add_module(
        self, node: ast.Module, name: str, *, 
        is_package: bool = False, 
        filename: Optional[str]=None
    ) -> "Mod":
        """
        Add a module to the project, all module should be added before calling `analyze_project`.

        :param node: Parsed `ast.Module` instance, see `ast.parse`.
        :param name: The fully qualified name of the module.
        :param is_package: Whether the module is a package (the node represents ``__init__.py`` file)
        :param filename: The filename of the module or ``__init__.py`` file for packages.
        """
        return cast(MutableState, self.state).add_module(
            node, name, is_package=is_package, filename=filename
        )

    def add_typeshed_module(self, modname: str) -> "Mod|None":
        return cast(MutableState, self.state).add_typeshed_module(modname)

    # TODO: introduce a generic reporter object used by System.msg, Documentable.report and here.
    def msg(self, msg: str, ctx: Optional[ast.AST] = None, thresh: int = 0) -> None:
        """
        Log a message about this ast node.
        """
        if self.options.verbosity < thresh:
            return
        context = ""
        if ctx:
            if isinstance(ctx, Def):
                ctx = ctx.node
            filename = self.state.get_filename(ctx)
            lineno = getattr(ctx, "lineno", "?")
            col_offset = getattr(ctx, "col_offset", None)
            if col_offset:
                context = f"{filename or '<unknown>'}:{lineno}:{col_offset}: "
            else:
                context = f"{filename or '<unknown>'}:{lineno}: "

        print(f"{context}{msg}", file=self.options.outstream)
