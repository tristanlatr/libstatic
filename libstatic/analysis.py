from collections import defaultdict
import enum
from functools import partial
import sys
import time
from typing import (Any, Callable, Collection, Dict, Iterable, List, Mapping, NoReturn, Optional, Set, Tuple, Type, Union, 
                    TypeAlias,)

import gast as ast
import attr

from beniget.beniget import DefUseChains, Def, Ancestors
from beniget.beniget import ordered_set

from typeshed_client import get_stub_file
from typeshed_client.finder import parse_stub_file

from ast2json import ast2json
from json2ast import json2ast

import diskcache
import appdirs

from libstatic.astutils import get_context, Context, node2dottedname, op2func, iterassignfull, ast_repr
from libstatic.transform import Transform

# TODOs:
# - proper logging system. 
# - custom exceptions instead of ValueError.
# - integrate transformations when add_module() if called.
#
# 

print = partial(print, flush=True)

def _load_typeshed_mod_spec(modname:str) -> Tuple[str, ast.Module, bool]:
    path = get_stub_file(modname)
    if not path:
        raise ValueError(f'module {modname} not found')
    is_package = path.stem == '__init__'
    return modname, ast.ast_to_gast(parse_stub_file(path)), is_package

class AbstractNodeVisitor:
    def generic_visit(self, node: ast.AST, *args:Any, **kwargs:Any) -> Any:
        raise ValueError(f'unsupported node type: {type(node).__name__}')
    
    def visit(self, node: ast.AST, *args:Any, **kwargs:Any) -> Any:
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

class NodeVisitor(AbstractNodeVisitor):
    def generic_visit(self, node: Any, *args: Any, **kwargs: Any) -> None:
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item, *args, **kwargs)
            elif isinstance(value, ast.AST):
                self.visit(value, *args, **kwargs)

ModuleName:TypeAlias = Tuple[str, ...]

def get_mod_name(name:Union[str, ModuleName]) -> ModuleName:
    """
    Coerce a dotted name string into a tuple of strings.
    """
    if isinstance(name, tuple):
        return name 
    elif isinstance(name, str):
        return tuple(name.split('.'))
    else:
        raise TypeError(f'name should be str or tuple, not {type(name)}')

_NAMED_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, 
                ast.AsyncFunctionDef, ast.Name, ast.alias)

@attr.s(auto_attribs=True)
class Options:
    python_version:Optional[Tuple[int, int]]=None
    platform:Optional[str]=None
    nested_dependencies:int=0

class State:
    """
    The `Project`'s state.
    """

    def __init__(self) -> None:
        self._modules: Dict[ModuleName, 'Mod'] = {}
        """
        Mapping from module names to Mod instances.
        """

        self._unreachable: Set[ast.stmt] = set()
        """
        Set of unreachable nodes.
        """

        self._locals: Dict[ast.AST, Dict[str, List[Def]]] = {}
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
        
        self._imported_names: Dict[ast.alias, ImportedName] = {}
        """
        Mapping from alias nodes to their parsed ImportedName.
        """

        self._modules__all__:Mapping['Mod', Tuple['Collection[str]|None', Collection[str]]] = {}
        """
        Mapping from Mod instances to tuple: (explicit ``__all__``, implicit public names).
        """
    
        
    def get_def(self, node:'ast.AST') -> Union[Def, 'Mod']:
        """
        Def-Use chains accessor.
        """
        try:
            return self._def_use_chains[node]
        except KeyError as e:
            raise ValueError(f'node {ast_repr(node)} is not present in the def-use chains') from e
    
    def get_imported_name(self, node:ast.alias) -> 'ImportedName':
        """
        Returns the parsed `ImportedName` for the given ast.alias instance.
        """
        try:
            return self._imported_names[node]
        except KeyError as e:
            raise ValueError(f'{ast_repr(node)} is not present in the imported names') from e
    
    def goto_def(self, node:ast.AST) -> Def:
        """
        Use-Def chains accessor that returns only one def, or raise ValueError.
        """
        return self.goto_defs(node)[0]
    
    def goto_defs(self, node:ast.AST, noraise:bool=False) -> List[Def]:
        """
        Use-Def chains accessor. It does not work for builtins at the moment.
        """
        try:
            defs = self._use_def_chains[node]
            if not defs:
                raise KeyError('empty use-def')
            return defs
        except KeyError as e:
            if not noraise:
                raise LookupError(f'no Def found for node {ast_repr(node)}') from e
            else:
                return []

    def get_module(self, name:Union[str, Tuple[str,...]]) -> Optional['Mod']:
        """
        Returns the module with the given name.
        """
        return self._modules.get(get_mod_name(name))

    def get_all_modules(self) -> Iterable['Mod']:
        return self._modules.values()

    def get_sub_module(self, mod:Union['Mod', ast.Module], name:str,) -> Optional['Mod']:
        if isinstance(mod, ast.AST):
            mod = self.get_def(mod)
        return self._modules.get(get_mod_name(mod.name())+(name,))

    def get_locals(self, node:Union['Mod', Def, ast.AST]) -> Mapping[str, List[Def]]:
        if isinstance(node, Def):
            node = node.node
        return self._locals[node]

    def get_local(self, node: Union['Mod', Def, ast.AST], name:str) -> List[Def]:
        try:
            return self.get_locals(node)[name]
        except KeyError:
            return []
        
    def get_attribute(self, namespace: Union['Mod', Def, ast.AST], name:str, *, 
                    ignore_locals:bool=False) -> List[Union['Mod', Def]]:
        """
        Get local attributes definitions matching the name from this scope.
        It calls both `get_local()` and `get_sub_module()`.
        """
        # TODO: Handle {"__name__", "__doc__", "__file__", "__path__", "__package__"}
        # TODO: Handle {__class__, __module__, __qualname__}
        # TODO: Handle instance variables
        # TODO: Handle looking up in super classes

        if isinstance(namespace, ast.AST):
            namespace = self.get_def(namespace)
        if not ignore_locals:
            values = self.get_local(namespace, name)
        else: 
            values = []
        if not values and isinstance(namespace, Mod) and namespace.is_package:
            # Support for sub-packages.
            sub = self.get_sub_module(namespace, name)
            if sub:
                return [sub]     
        if values:
            return values
        raise ValueError(f'{name!r} not found in {namespace}')
    
    def get_fullname(self, stmt:ast.AST) -> str:
        """
        Get the fully qualified name of the definition coresponding to the given ast node.

        :raises TypeError: If the definition is not a named node. 
        """
        if not isinstance(stmt, _NAMED_NODES):
            raise TypeError(f'{stmt.__class__.__name__} is not a named node, named nodes are {[t.__name__ for t in _NAMED_NODES]}')
        named_nodes = [node for node in self.get_parents(stmt)+[stmt] if \
                       isinstance(node, _NAMED_NODES)]
        names = [self.get_def(node).name() for node in named_nodes]
        return '.'.join(names)

    def get_dunder_all(self, mod:'Mod') -> 'Collection[str]|None':
        try:
            return self._modules__all__[mod][0]
        except KeyError as e:
            raise ValueError() from e
    
    def get_public_names(self, mod:'Mod') -> Collection[str]:
        try:
            __all__, implicit__all__ = self._modules__all__[mod]
            if __all__ is not None:
                return __all__
            return implicit__all__
        except KeyError as e:
            raise ValueError() from e
    
    def get_parent(self, node:ast.AST) -> ast.AST:
        return self._ancestors[node][-1]

    def get_parents(self, node:ast.AST) -> List[ast.AST]:
        return self._ancestors[node]

    def get_parent_instance(self, node:ast.AST, cls:'Type[ast.AST]|Tuple[Type[ast.AST],...]') -> ast.AST:
        for n in reversed(self._ancestors[node]):
            if isinstance(n, cls):
                return n
        raise ValueError("{} has no parent of type {}".format(node, cls))

    def is_reachable(self, node:ast.AST) -> bool:
        return node not in self._unreachable

    def dump(self) -> 'list[dict[str, Any]]':
        def _dump_mod(_m:Mod) -> 'dict[str, Any]':
            return {
                'is_package':_m.is_package,
                'modname':_m.name(),
                'node':ast2json(ast.gast_to_ast(_m.node))
            }
        return [_dump_mod(m) for m in self._modules.values()]

class StateModifiers:
    """
    Class wrapping modifiers for the `State`.

    Among others, it that ensures that modifications of the def-use chains 
    are replicated in the use-def chains as well, as if they form a unique
    data structure.
    """

    # State modifiers/setters are implemented in their own class.

    def __init__(self, state:State) -> None:
        self._state = state
    
    def add_typeshed_module(self, modname:str) -> 'Mod|None':
        try:
            _, modast, is_pack = _load_typeshed_mod_spec(modname)
        except ValueError:
            print(f'module not found {modname}')
            return None
        new_mod = self.add_module(modast, modname, is_package=is_pack)
        return new_mod

    def add_module(self, node: ast.Module, name:str, *, is_package:bool=False) -> 'Mod':
        """
        Adds a module to the project. 
        All modules should be added before calling `analyze_project()`.
        This will transform the AST so it's compatible with libstatic.
        """

        mod = Mod(Transform().transform(node), name, is_package=is_package)
        # add module to the chains
        self._state._def_use_chains[node] = mod
        oldmod = self._state._modules.setdefault(get_mod_name(mod.name()), mod)
        if mod is not oldmod:
            return self.handle_duplicate_module(mod, oldmod)
        else:
            return mod
    
    def handle_duplicate_module(self, mod:'Mod', oldmod:'Mod') -> 'Mod':
        return mod

    # use-def-use structure
    
    def _add_usedef(self, use:Def, definition:Def) -> None:
        self._state._use_def_chains.setdefault(use.node, []).append(definition)
    
    def add_definition(self, definition:Def) -> None:
        assert definition.node not in self._state._def_use_chains
        self._state._def_use_chains[definition.node] = definition
        for u in definition.users():
            self._add_usedef(u, definition)

    def add_user(self, definition:Def, use:Def) -> None:
        definition.add_user(use)
        self._add_usedef(use, definition)

    def remove_user(self, definition:Def, use:Def) -> None:
        definition._users.discard(use)
        self._state._use_def_chains[use.node].remove(definition)
    
    def remove_definition(self, definition:Def) -> None:
        del self._state._def_use_chains[definition.node]
        for use in definition.users():
            self.remove_user(definition, use)
    
    # first pass updates

    def _update_defuse(self, defuse:Dict[ast.AST, Def]) -> None:
        self._state._def_use_chains.update(defuse)
    
    def _update_locals(self, locals:Dict[ast.AST, Def]) -> None:
        self._state._locals.update(locals)

    def _update_ancestors(self, ancestors:Dict[ast.AST, List[ast.AST]]) -> None:
        self._state._ancestors.update(ancestors)
    
    def _update_usedef(self, usedef:Dict[ast.AST, List[Def]]) -> None:
        self._state._use_def_chains.update(usedef)
    
    def _update_imports(self, imports:Mapping[ast.alias, 'ImportedName']) -> None:
        self._state._imported_names.update(imports)

    def _update_unreachable(self, unreachable:Set[ast.AST]) -> None:
        self._state._unreachable.update(unreachable)

    def store_anaysis(self, *, defuse:'Dict[ast.AST, Def]|None'=None,
                      locals:'Dict[ast.AST, Def]|None'=None,
                      ancestors:'Dict[ast.AST, List[ast.AST]]|None'=None,
                      usedef:'Dict[ast.AST, List[Def]]|None'=None,
                      imports:'Mapping[ast.alias, ImportedName]|None'=None, 
                      unreachable:'Set[ast.AST]|None'=None,) -> None:
        self._update_defuse(defuse) if defuse else...
        self._update_locals(locals) if locals else...
        self._update_ancestors(ancestors) if ancestors else...
        self._update_usedef(usedef) if usedef else...
        self._update_imports(imports) if imports else...
        self._update_unreachable(unreachable) if unreachable else...

    # loading 

    def load(self, data:'list[dict[str, Any]]') -> None:
        for mod_spec in data:
            assert all(k in mod_spec for k in ['node', 'modname', 'is_package'])
            self.add_module(ast.ast_to_gast(json2ast(mod_spec['node'])), 
                           mod_spec['modname'], 
                           is_package=mod_spec['is_package'])   

class Analyzer:
    def __init__(self, state:State, options:Options) -> None:
        self._options = options
        self._state = state
        self._stmod = StateModifiers(state)
    
    def _analyze_module_pass1(self, mod:'Mod') -> None:
        module_node = mod.node
        # : Accumulate static analysis infos from beniget
        # - compute local def-use chains
        defuse, locals = DefUseChainsAndLocals(module_node, filename=mod.name())
        # - add the locals collected by beniget
        
        # - compute ancestors
        ancestors_vis = Ancestors()
        ancestors_vis.visit(module_node)
        ancestors = ancestors_vis._parents

        usedef = UseDefChains(defuse)
        
        # : Parsing imports
        imports = ParseImportedNames(mod.name(), is_package=mod.is_package
                                ).visit_Module(module_node)
        
        self._stmod.store_anaysis(defuse=defuse, locals=locals,
                                  ancestors=ancestors, usedef=usedef, imports=imports)
        
        # : Reachability analysis
        unreachable = Unreachable(self._state, self._options, module_node)

        self._stmod.store_anaysis(unreachable=unreachable)

    def _analyzer_pass1(self) -> None:

        processed_modules = set()
        to_process = [mod.name() for mod in self._state.get_all_modules()]
        iteration = 0
        
        t0 = time.time()
        
        while to_process:
            for name in list(to_process):
                if name not in processed_modules:
                    
                    mod = self._state.get_module(name)
                    if not mod:
                        # a dependency module
                        try:
                            _, modast, is_pack = _load_typeshed_mod_spec(name)
                        except ValueError:
                            print(f'module not found {name}')
                            continue
                        mod = self._stmod.add_module(modast, name, is_package=is_pack)
                    
                    self._analyze_module_pass1(mod)
                    
                    # add dependencies
                    if iteration!=self._options.nested_dependencies:
                        for imp in (self._state.get_imported_name(al) for al in 
                                    (n for n in ast.walk(mod.node) 
                                     if isinstance(n, ast.alias))):
                            to_process.append('.'.join(imp.orgmodule))
                to_process.remove(name)
                processed_modules.add(name)
            iteration += 1
        
        t1= time.time()
        print(f'dependency loading took {t1-t0} seconds')

    def analyze(self) -> None:
        """
        Initiate the project state. 
        """
        t0 = time.time()
        
        self._analyzer_pass1()

        # : Imports analysis: complement def-use chains with import chains
        # must be done after all modules have been added
        for mod in self._state.get_all_modules():
            ChainDefUseOfImports(self._state).visit(mod.node)
        
        # at this point goto definition is working, for non wildcard imports names
        
        # : Compute __all__ and wildcard imports and fixup def-use chains for wildcard imports 
        self._state._modules__all__ = ComputeWildcards(self._state)
        
        # : sort out what's an api object and what's not
        # : resolve class mros
        # : calculate instance attributes
        # : calculate the canonical full name for every api objects

        t1= time.time()
        print(f'analysis took {t1-t0} seconds')

class Project:
    """
    A project is a the higth level wrapper to analyze ast modules together.

    Assumptions made by this class that developer should be aware of: 
    - Augmented assignments have been transformed into regular assignments.
    - All modules are added before caling `analyze_project()`
    - No other modules are added after caling `analyze_project()`.
    - Many assumptions regarding ``__all__`` module variables:
      - The only dynamic value present in the list can be other module's __all__
      - Building __all__ canot be done inside a loop: it must only use regular assignemnts,
        ``__all__.append()`` and ``__all__.extend()`` can be refactored into regular 
        assignments by a node transformer.
    """
    def __init__(self, **kw:Any) -> None:
        """
        :param kw: All parameters are passed to `Options` constructor.
        """
        self.options = Options(**kw)
        self.state = State()
    
    def analyze_project(self) -> None:
        Analyzer(self.state, self.options).analyze()
    
    def add_module(self, node: ast.Module, name:str, *, is_package:bool=False) -> 'Mod':
        return StateModifiers(self.state).add_module(node, name, is_package=is_package)

class Mod(Def):
    """
    Model a python module.
    Interface is designed to be as minimalistic as possible as well as integrated into the
    wider use-def chains.
    """

    __slots__ = "node", "_users", "_modname", "is_package"
    node: ast.Module
    
    def __init__(self, node:ast.Module, modname:str, is_package:bool=False) -> None:
        super().__init__(node)
        self._modname = modname
        self.is_package = is_package
    
    def name(self) -> str:
        return self._modname

# several names can be imported from the same import statement, 
# and different priority might be added depeding on the name of the import.
# we need to thin-down the ast to have only one name per instance of the model here
class ImportedName:
    """
    Wraps L{ast.Import} or L{ast.ImportFrom} nodes.
    
    @note: One L{ImportedName} instance is created for each 
        name bound in the C{import} statement.
    """
    __slots__ = 'node', 'orgmodule', 'orgname', 'fullorgmodule'
    
    def __init__(self, node:ast.alias, orgmodule:'ModuleName|str', 
                 orgname:'str|None'=None, 
                 fullorgmodule:'ModuleName|str|None'=None) -> None:
        self.node = node
        self.orgmodule = get_mod_name(orgmodule)
        self.orgname = orgname
        if not fullorgmodule:
            self.fullorgmodule = self.orgmodule
        else:
            self.fullorgmodule = get_mod_name(fullorgmodule)
    
    def target(self) -> str:
        if self.orgname:
            return '.'.join((*self.orgmodule, self.orgname))
        else:
            return '.'.join(self.orgmodule)
    
    def name(self) -> str:
        return (self.node.asname or self.node.name).split(".", 1)[0]

class wildcard_imported_name(ast.alias):
    """
    A fictional ast node to represent a particular name wildcard imports are binding.
    The fact that it subclasses `ast.alias` is important because instances of `wildcard_imported_name`
    can be used anytime `ast.alias` is expected. 
    """

    def __init__(self, name:str, module:str, *, lineno:int, col_offset:int):
        super().__init__(name=name, asname=None, lineno=lineno, col_offset=col_offset)
        self.module = module

def DefUseChainsAndLocals(node:ast.Module, filename:Any=None,) -> Tuple[Dict[ast.AST, Def], 
                                                                        Dict[ast.AST, Dict[str, List[Def]]]]:
    
    # - compute local def-use chains
    defuse = DefUseChains(filename=filename)
    assert hasattr(defuse, 'future_annotations')
    defuse.future_annotations = True
    defuse.visit(node)
    locals_as_dict: Dict[ast.AST, Dict[str, List[Def]]] = {}
    for namespace,loc_list in defuse.locals.items():
        d = locals_as_dict.setdefault(namespace, {})
        for loc in loc_list:
            d.setdefault(loc.name(), []).append(loc)
    return defuse.chains, locals_as_dict

def UseDefChains(def_use_chains: Dict[ast.AST, Def]) -> Dict[ast.AST, List[Def]]:
    """
    Flip the Def-Use chains to generate Use-Def chains. It does not include the use of buitins.
    """
    chains:Dict[ast.AST, List[Def]] = {}
    for chain in def_use_chains.values():
        # init a empty list for all Name and alias instances
        if isinstance(chain.node, (ast.Name, ast.alias)):
            chains.setdefault(chain.node, [])
        
        for use in chain.users():
            chains.setdefault(use.node, []).append(chain)
    
    return chains
    # this does not support builtins, by design

def Unreachable(state: State, options:Options, mod:ast.Module) -> Set[ast.AST]:

    known_values: Dict[str, Any] = {}
    version = options.python_version
    if version:
        assert isinstance(version, tuple)
        assert len(version)>=2
        assert all(isinstance(p, int) for p in version)

        known_values['sys.version_info'] = version
        known_values['sys.version_info.major'] = version[0]
        known_values['sys.version_info.minor'] = version[1]
    
    return _Unreachable(state, known_values).visit_Module(mod)

class _Unreachable(NodeVisitor):

    def __init__(self, state:State, known_values:Mapping[str, Any]) -> None:
        self._state = state
        self._known_values: Mapping[str, Any] = known_values
        self._unreachable_nodes: Set[ast.AST] = set()
    
    def visit_If(self, node:ast.If) -> None:
        try:
            testval = LiteralEval(self._state, node.test, 
                                    known_values=self._known_values)
        except Exception as e:
            self.generic_visit(node)
        else:
            unreachable = node.orelse if testval else node.body
            reachable = node.body if testval else node.orelse
            mark_unreachable = _MarkUnreachable(self._unreachable_nodes)
            for n in unreachable:
                print(f'{ast_repr(n)} is unreachable.')
                mark_unreachable.visit(n)
            for n in reachable:
                self.generic_visit(n)
    
    def visit_Module(self, node:ast.Module) -> Set[ast.AST]:
        self.generic_visit(node)
        return self._unreachable_nodes

class _MarkUnreachable(NodeVisitor):
    
    def __init__(self, unreachable:'set[ast.AST]') -> None:
        self._unreachable = unreachable
    
    def visit(self, node: ast.AST) -> Any: #type:ignore[override]
        self._unreachable.add(node)
        self.generic_visit(node)

class ImportParser(NodeVisitor):
    """
    Transform import statements into a series of `ImportedName`s.

    One instance of ImportParser can be used to parse all imports in a given module.
    """
    
    # refactor: to arguments modname:str and is_package:bool
    def __init__(self, modname:'ModuleName|str', *, is_package:bool) -> None:
        self._modname = get_mod_name(modname)
        self._is_package = is_package
        self._result:List[ImportedName] = []
    
    def generic_visit(self, node: Any) -> None: #type:ignore[override]
        AbstractNodeVisitor.generic_visit(self, node)
    
    # parsing imports, partially adjusted from typeshed_client

    def visit_Import(self, node: ast.Import) -> List[ImportedName]:
        self._result.clear()
        for al in node.names:
            if al.asname:
                self._result.append(
                    ImportedName(al, orgmodule=al.name))
            else:
                # here, we're lossng the dependency on "driver" of "import pydoctor.driver" in 'orgmodule',
                # but we can use fullorgmodule to retreive it anyway.
                self._result.append(
                    ImportedName(al, orgmodule=al.name.split(".", 1)[0], 
                                 fullorgmodule=al.name))
        return self._result
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> List[ImportedName]:
        self._result.clear()
        current_module: Tuple[str, ...] = get_mod_name(self._modname)
        module: Tuple[str, ...]
        
        if node.module is None:
            module = ()
        else:
            module = tuple(node.module.split("."))
        if node.level == 0:
            source_module = module
        elif node.level == 1:
            if self._is_package:
                source_module = current_module + module
            else:
                source_module = current_module[:-1] + module
        else:
            if self._is_package:
                source_module = current_module[: 1 - node.level] + module
            else:
                source_module = current_module[: -node.level] + module
        for alias in node.names:
            if alias.name == "*":
                self._result.append(
                    ImportedName(alias, orgmodule=source_module, orgname='*')
                )
                # store the wildcard import and process it later...
            else:
                self._result.append(
                    ImportedName(alias, orgmodule=source_module, orgname=alias.name)
                )
        return self._result

class ParseImportedNames(NodeVisitor):
    """
    Maps each `ast.alias` to their `ImportedName` counterpart.
    """
    def __init__(self, modname:'ModuleName|str', *, is_package:bool) -> None:
        super().__init__()
        self._modname = get_mod_name(modname)
        self._is_package = is_package
        self._import_parser = ImportParser(modname, is_package=is_package)
    
    def visit_Module(self, node:ast.Module) -> Mapping[ast.alias, ImportedName]:
        self._result:Dict[ast.alias, ImportedName] = {}
        self.generic_visit(node)
        return self._result
    
    def generic_visit(self, node: Any) -> None:  #type:ignore[override]
        if isinstance(node, ast.expr):
            return
        else:
            super().generic_visit(node)
    
    def visit_Import(self, node:Union[ast.Import, ast.ImportFrom]) -> None:
        for name in self._import_parser.visit(node):
            self._result[name.node] = name
    
    visit_ImportFrom = visit_Import

class ChainDefUseOfImports(NodeVisitor):
    """
    Adds each alias instance to the list of uses of the Def of the name their are binding.
    """
    
    def __init__(self, state:State) -> None:
        self._state = state
    
    def generic_visit(self, node: Any) -> None:  #type:ignore[override]
        if isinstance(node, ast.expr):
            return
        else:
            super().generic_visit(node)

    def visit_Import(self, node:ast.Import) -> None:
        modifiers = StateModifiers(self._state)
        
        for alias in node.names:
            name = self._state.get_imported_name(alias)
            orgmodule = self._state.get_module(name.orgmodule)
            if orgmodule:
                orgname = name.orgname
                alias_def = self._state.get_def(name.node)
                if orgname:
                    if orgname=='*':
                        continue
                    try:
                        defs = self._state.get_attribute(orgmodule, orgname) # todo: handle ignore locals
                    except ValueError:
                        continue
                        # import target not found
                    else:
                        for loc in defs:
                            modifiers.add_user(loc, alias_def)
                else:
                    modifiers.add_user(orgmodule, alias_def)
            else:
                # module not found in the system
                continue
    
    visit_ImportFrom = visit_Import

def GetStoredValue(node: ast.AST, assign:'ast.Assign|ast.AnnAssign') -> Optional[ast.expr]:
    """
    Given an ast.Name instance with Store context and it's assignment statement, 
    figure out the right hand side expression that is stored in the symbol.

    Limitation: 
        - Starred assignments are currently not supported as they ususally mean
          we're unpacking something of variable lenght.
        - Nested tuple assignments are not supported.
        - For loops targets are not supported by this function, 
          it need to return an object that represent an expression of type T,
          not the expression itself, since it usually have multiple values.
    
    :raises ValueError: Whenever there is a assignment that we can't understand.
    """

    def _fail(_n:ast.AST) -> NoReturn:
        raise ValueError(f'unsupported assignment: {ast_repr(_n)}')
    
    # There is no augmented assignments
    value = assign.value
    is_sequence = isinstance(value, (ast.List, ast.Tuple))
    for _, target in iterassignfull(assign):
        if target is node:
            return value
        elif isinstance(target, (ast.List, ast.Tuple)):
            if is_sequence and len(target.elts)==len(value.elts):
                try:
                    index = target.elts.index(node)
                except IndexError:
                    continue
                else:
                    try:
                        element = value.elts[index]
                    except IndexError:
                        _fail(assign)
                    else:
                        if isinstance(element, ast.Starred):
                            _fail(assign)
                        return element
            else:
                _fail(assign)
    _fail(assign)

class _ProcessingState(enum.Enum):
    UNPROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2

class DeferredModuleException(Exception):
    """
    Raised when a module is part of cycle.
    """
    def __init__(self, modname:'ModuleName|str') -> None:
        super().__init__()
        self.modname = modname

class OrderedBuilder:

    def __init__(self, state: State, processModuleAST: Callable[[ast.Module], Any], max_iterations: int = 1) -> None:
        super().__init__()
        self._state = state
        self._processModuleAST = processModuleAST
        self._result: Dict[Mod, Any] = {}

        self._processing_state: Dict[Mod, _ProcessingState] = defaultdict(lambda: _ProcessingState.UNPROCESSED)
        self._unprocessed_modules = ordered_set(state.get_all_modules())
        self._processing_modules: List[str] = []
        self._max_iterations = max_iterations
        self.iteration = 0
    
    def is_last_iteration(self) -> bool:
        return self.iteration==self._max_iterations
    
    def is_processed(self, mod:'Mod|None') -> bool:
        if not mod: 
            return True
        return self._processing_state[mod] is _ProcessingState.PROCESSED

    def get_processed_module(self, modname: Union[str, Tuple[str, ...]]) -> Optional[Mod]:
        # might return a processing module in the case of cyclic imports
        print(f'needs module {".".join(get_mod_name(modname))}')
        mod = self._state.get_module(modname)
        if mod is None:
            return None
        if self._processing_state[mod] is _ProcessingState.PROCESSING:
            return mod
        if self._processing_state[mod] is _ProcessingState.UNPROCESSED:
            self._process_module(mod)
        return mod

    def get_processed_module_or_raise(self, modname: Union[str, Tuple[str, ...]]) -> Optional[Mod]:
        # will not raise if it's the last iteration
        mod = self.get_processed_module(modname)
        if not self.is_processed(mod):
            if not self.is_last_iteration():
                raise DeferredModuleException(modname)
        return mod

    def _process_module(self, mod: Mod) -> None:
        assert self._processing_state[mod] is _ProcessingState.UNPROCESSED, self._processing_state[mod]
        assert mod in self._unprocessed_modules
        self._processing_state[mod] = _ProcessingState.PROCESSING
        self._unprocessed_modules.values.pop(mod)
        self._processing_modules.append(mod.name())
        try:
            print(f'processing module {mod.name()}')
            self._result[mod] = self._processModuleAST(mod.node)
        except DeferredModuleException as e:
            print(f'deferring process of module {mod.name()} because {".".join(get_mod_name(e.modname))} is not processed yet')
            target_mod = self._state.get_module(e.modname)
            if target_mod:
                # set both modules to unprocessed
                self._processing_state[mod] = self._processing_state[target_mod] = _ProcessingState.UNPROCESSED
                self._unprocessed_modules.update((target_mod, mod,))
            else:
                self._processing_state[mod] = _ProcessingState.UNPROCESSED
                self._unprocessed_modules.update((mod,))

            head = self._processing_modules.pop()
            assert head == mod.name()
        else:
            head = self._processing_modules.pop()
            assert head == mod.name()
        
        if self._processing_state[mod] is not _ProcessingState.UNPROCESSED:
            self._processing_state[mod] = _ProcessingState.PROCESSED
    
    def build(self) -> Dict[Mod, Any]:
        assert self.iteration == 0
        while self._unprocessed_modules:
            if self.iteration < (self._max_iterations or 1):
                for mod in list(self._unprocessed_modules):
                    if self._processing_state[mod] is _ProcessingState.UNPROCESSED:
                        self._process_module(mod)
                self.iteration+=1
            else:
                raise ValueError(f'Could not process modules: {[m.name() for m in self._unprocessed_modules]}. '
                                 'No DeferredModuleException exceptions should be raised during the last iteration')
        return self._result

# could be refactored into a Project method.
def ExpandName(state:State, node: ast.expr) -> 'str|None':
    """
    Resove expression composed by L{ast.Attribute} and L{ast.Name} nodes to it's possibly fully qualified name
    by using information only availlable in the current module.
    
    '''
    from twisted.web.template import Tag as TagType
    v = TagType() # <- expanded name is 'twisted.web.template.Tag', even if Tag is actually imported from another module.
    '''
    """

    dottedname = node2dottedname(node)
    if not dottedname:
        return None
    # If node2dottedname() returns something, the expression is composed by one Name
    # and potentially multiple Attribute instance. So the following line is safe.
    top_level_name = next(name for name in ast.walk(node) if isinstance(name, ast.Name))
    try:
        def_node = state.goto_def(top_level_name).node
    except ValueError:
        # unbound name
        return None
    if isinstance(def_node, ast.alias):
        imported_name = state.get_imported_name(def_node)
        dottedname[0] = imported_name.target()
        return '.'.join(dottedname)
    else:
        return state.get_fullname(def_node)

_AST_SEQUENCE_TO_TYPE = {
    ast.Tuple: tuple,
    ast.List: list, 
    ast.Set: set
}

class _ASTEval(NodeVisitor):

    _MAX_JUMPS = int(sys.getrecursionlimit()/1.5)
    # each visited ast node counts for one jump.

    def __init__(self, state: State, 
                 known_values: Mapping[str, Any]) -> None:
        self._state = state
        self._known_values = known_values
        super().__init__()
    
    def generic_visit(self, node: Any, *args: Any, **kwargs: Any) -> None:
        # fails on unknown nodes
        AbstractNodeVisitor.generic_visit(self, node, *args, **kwargs)
    
    def _returns(self, ob:ast.stmt, path:List[ast.AST]) -> Def:
        return ob
    
    visit_Module = \
        visit_ClassDef = \
        visit_FunctionDef = \
        visit_AsyncFunctionDef = \
        visit_Lambda = _returns
    
    def visit(self, node: Any, path:List[ast.AST]) -> Any: # type:ignore[override]
        if node in path:
            raise ValueError('cyclic definition')
        if len(path) > self._MAX_JUMPS:
            raise ValueError('expression is too complex')
        # fork path
        path = path.copy()
        path.append(node)
        return super().visit(node, path)

    def visit_Attribute(self, node:ast.Attribute, path: List[ast.AST]) -> Any:
        if get_context(node) is not Context.Load:
            raise ValueError(f'unexpected context for node {node}: {get_context(node)}')
        # check if this name is part of the known values
        fullname = ExpandName(self._state, node)
        if fullname in self._known_values:
            return self._known_values[fullname] # type:ignore[index]
        
        namespace = self.visit(node.value, path)
        if isinstance(namespace, (ast.Module, ast.ClassDef)):
            attrib = self._state.get_attribute(namespace, node.attr)[0]
            return self.visit(attrib.node, path)
        else:
            raise ValueError(f'getattr not supported on value {namespace}')

    def visit_alias(self, node:ast.alias, path: List[ast.AST]) -> Any:
        fullname = self._state.get_imported_name(node).target()
        if fullname in self._known_values:
            return self._known_values[fullname]
        raise ValueError(f'unknown value: {fullname}')
    
    # wildcard_imported_name
    visit_wildcard_imported_name = visit_alias
    
    def visit_Name(self, node:ast.Name, path:List[ast.AST]) -> Any:
        ctx = get_context(node)
        if ctx is Context.Load:
            # TODO: integrate with reachability analysis
            
            # check if this name is part of the known values
            if node.id in self._known_values:
                return self._known_values[node.id]

            # Use goto to compute the value of this symbol
            name_def = self._state.goto_def(node)
            return self.visit(name_def.node, path)
        
        elif ctx is Context.Store:
            # we live in a world where no augmented assignments exists, so that's good.
            try:
                assign = self._state.get_parent_instance(node, (ast.Assign, ast.AnnAssign))
            except ValueError as e:
                raise ValueError(f'unsupported name: {ast.dump(node)}') from e
            value = GetStoredValue(node, assign=assign)
            if value is not None:
                return self.visit(value, path)
            else:
                raise ValueError(f'no value for {ast.dump(node)}')
        
        else:
            raise ValueError(f'unsupported ctx={ctx} for {ast.dump(node)}')

class _LiteralEval(_ASTEval):

    def visit_Constant(self, node:ast.Attribute, path: List[ast.AST]) -> Any:
        return node.value

    def visit_List(self, node:Union[ast.Tuple, ast.List, ast.Set], 
                   path:List[ast.AST]) -> Any:
        if get_context(node) is not Context.Load:
            raise ValueError(f'unexpected context for node {node}: {get_context(node)}')
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                values.extend(self.visit(elt.value, path))
            else:
                values.append(self.visit(elt, path))
        try:
            return _AST_SEQUENCE_TO_TYPE[type(node)](values)
        except TypeError as e:
            raise ValueError('invalid container') from e

    visit_Tuple = visit_List
    visit_Set = visit_List

    def visit_BinOp(self, node:ast.BinOp, path:List[ast.AST]) -> Any:
        return op2func(node.op)(self.visit(node.left, path),
                                self.visit(node.right, path))

    def visit_BoolOp(self, node:ast.BoolOp, path:List[ast.AST]) -> Any:
        for val_node in node.values:
            val = self.visit(val_node, path)
            if (isinstance(node.op, ast.Or) and val) or (
                isinstance(node.op, ast.And) and not val
            ):
                return val
        return val
    
    def visit_Compare(self, node:ast.Compare, path:List[ast.AST]) -> Any:
        """comparison operators, including chained comparisons (a<b<c)"""
        lval = self.visit(node.left, path)
        results = []
        for oper, rnode in zip(node.ops, node.comparators):
            rval = self.visit(rnode, path)
            ret = op2func(oper)(lval, rval)
            results.append(ret)
            lval = rval
        if len(results) == 1:
            return results[0]
        out = True
        for ret in results:
            out = out and ret
        return out
    
    def visit_Subscript(self, node: ast.Subscript, path:List[ast.AST]) -> Any:
        if get_context(node) is not Context.Load:
            raise ValueError(f'unexpected context for node {node}: {get_context(node)}')
        value = self.visit(node.value, path)
        slc = self.visit(node.slice, path)
        return value[slc]

    def visit_Slice(self, node: ast.Slice, path:List[ast.AST]) -> slice:
        lower = self.visit(node.lower, path) if node.lower is not None else None
        upper = self.visit(node.upper, path) if node.upper is not None else None
        step = self.visit(node.step, path) if node.step is not None else None
        return slice(lower, upper, step)
    
    def visit_NamedExpr(self, node:ast.NamedExpr, path:List[ast.AST]) -> Any:
        return self.visit(node.value, path)

class _GotoDefinition(_ASTEval):
    def __init__(self, state: State) -> None:
        super().__init__(state, {})
    
    def visit_alias(self, node:ast.alias, path: List[ast.AST]) -> Any:
        return self.visit(self._state.goto_def(node).node, path)
    
    def visit_Name(self, node: ast.Name, path: List[ast.AST]) -> Any:
        ctx = get_context(node)
        if ctx is Context.Load:
            return super().visit_Name(node, path)
        else:
            return node

def LiteralEval(state:State, node:ast.expr, *, known_values:'Mapping[str, Any]|None'=None) -> Any:
    """
    Powerfull ``ast.literal_eval()`` function.
    """
    visitor = _LiteralEval(state, known_values=known_values or {})
    result = visitor.visit(node, [])
    if isinstance(result, ast.AST):
        raise ValueError(f'No literal value for node {node}, got {type(result).__name__}')
    return result

def GotoDefinition(state:State, node:ast.expr) -> Def:
    return state.get_def(_GotoDefinition(state).visit(node, []))

class _VisitDunderAllAssignment(NodeVisitor):
    """
    Ensures that dependencies required to calculate __all__ are processed before 
    going forward. Currently, only other __all__ values will be considered as dependencies
    to calculate __all__.
    """
    def  __init__(self, state: State, builder:OrderedBuilder) -> None:
        self._state = state
        self._builder = builder

    def visit_Name(self, node:ast.Name) -> None:
        if get_context(node) is Context.Load:
            for d in self._state.goto_defs(node, noraise=True):
                if isinstance(d.node, ast.alias):
                    imported = self._state.get_imported_name(d.node)
                    if imported.orgname == '__all__':
                        self._builder.get_processed_module_or_raise(imported.orgmodule)

    def visit_Attribute(self, node:ast.Attribute) -> None:
        if get_context(node) is Context.Load and node.attr == '__all__':
            modulename = node.value
            fullname = ExpandName(self._state, modulename)
            if fullname:
                self._builder.get_processed_module_or_raise(fullname)

class _VisitWildcardImports(NodeVisitor):

    def  __init__(self, state: State, builder:OrderedBuilder) -> None:
        self._state = state
        self._builder = builder
        self._builder._result: Dict[Mod, Tuple['Collection[str] | None', 'Collection[str]']]
        self._result: Dict[ast.alias, Optional['Collection[str]']] = {}

    def visit_Module(self, node:ast.Module) -> Dict[ast.alias, Optional['Collection[str]']]:
        self.generic_visit(node)
        return self._result

    def visit_alias(self, node:ast.alias) -> None:
        if node.name == "*":
            imp = self._state.get_imported_name(node)
            mod = self._builder.get_processed_module_or_raise(imp.orgmodule)
            if mod:
                explicit_all, implicit_all = self._builder._result.get(mod, (None, None))
                expanded_wildcard = explicit_all if explicit_all is not None else implicit_all
                if expanded_wildcard is None:
                    ...
                    # this is the last iteration, so calculate it as a fallback
                self._result[node] = expanded_wildcard
            else:
                # not in the system
                self._result[node] = None
    
    def _returns(self, ob:ast.stmt) -> None:
        return

    visit_ClassDef = \
        visit_FunctionDef = \
        visit_AsyncFunctionDef = \
        visit_Lambda = _returns

def _compute_public_names(state:State, mod:ast.Module) -> Collection[str]:
    """
    In the absence of definition of __all__, we use this function to compute the names in a wildcard import.
    """
    return list(dict.fromkeys((n for n in state.get_locals(mod).keys()
                            if not n.startswith('_') and n != '*')))

class _ComputeWildcards(NodeVisitor):

    def  __init__(self, state: State, builder:OrderedBuilder) -> None:
        self._state = state
        self._builder = builder
        self._builder._result: Dict[Mod, Tuple['Collection[str] | None', 'Collection[str]']]
    
    def _process_wildcard_imports(self, node:ast.Module) -> None:
        visitor = _VisitWildcardImports(self._state, self._builder)
        alias2bnames = visitor.visit_Module(node)
        print(f'collected {len(alias2bnames)} wildcards from module {self._state.get_def(node).name()}')

        # Create Defs for each bounded names,
        # adding new defs that replaces the wildcard, 
        # this is a in-place modification to our model.
        
        modifiers = StateModifiers(self._state)
        
        for alias, bnames in reversed(alias2bnames.items()):
            if not bnames:
                print(f'wildcard could not be resolved: {ast_repr(alias)}')
                continue
            print(f'wildcard contains names: {bnames}')
            imp = self._state.get_imported_name(alias)
            old_def = self._state.get_def(alias)
            modifiers.remove_definition(old_def)
            # for each bounded names, replaces it's usages by 
            # a special Def that represent a specific name.
            for name in bnames:
                new_node = wildcard_imported_name(name, 
                                                  '.'.join(imp.orgmodule), 
                                                  lineno=alias.lineno, 
                                                  col_offset=alias.col_offset)
                resolved_def = Def(new_node)
                modifiers.add_definition(resolved_def)
                # We should use the modifiers for the following two lines
                self._state._locals[node].setdefault(name, []).append(resolved_def)
                self._state._imported_names[new_node] = ImportedName(new_node, 
                                    orgmodule=imp.orgmodule, 
                                    orgname=name)
                
                for unbound_name in list(old_def.users()):
                    if unbound_name.name()==resolved_def.name():
                        # cleanup (needed) over-approximations of beniget
                        for d in self._state.goto_defs(unbound_name.node):
                            modifiers.remove_user(d, unbound_name)
                        modifiers.add_user(resolved_def, unbound_name)

    def _process__all__dependencies(self, node:ast.Module) -> bool:
        # returns whether __all__ is defined in the module
        
        has_value = False
        for local__all__ in self._state.get_local(node, '__all__'):
            
            if isinstance(local__all__.node, ast.alias):
                # __all__ is an import
                imported = self._state.get_imported_name(local__all__.node)
                self._builder.get_processed_module_or_raise(imported.orgmodule)
                has_value = True
            else:
                # __all__ is defined but not an import
                try:
                    # __all__ is an assignment
                    assign = self._state.get_parent_instance(local__all__.node, 
                                                                    (ast.Assign, ast.AnnAssign))
                    value = GetStoredValue(local__all__.node, assign)
                    if value:
                        _VisitDunderAllAssignment(self._state, self._builder).visit(value)
                        has_value = True
                
                except ValueError:
                    # __all__ is defined but we can't figure out it's value    
                    pass
        
        return has_value
    
    def visit_Module(self, node: ast.Module) -> Tuple['Collection[str] | None', 'Collection[str]']:

        self._process_wildcard_imports(node)        
        implicit__all__ = _compute_public_names(self._state, node)
        explicit__all__ = None

        if self._process__all__dependencies(node):
            __all__def = self._state.get_local(node, '__all__')[-1]
            known_values = {f'{mod.name()}.__all__':_all \
                            for mod, (_all, _) in self._builder._result.items() if \
                                isinstance(_all, (tuple, list))}
            try:
                explicit__all__ = LiteralEval(self._state, __all__def.node, 
                                                known_values=known_values)
            except (TypeError, ValueError) as e:
                print(f'cannot compute value of {ast_repr(__all__def.node)}, got {e.__class__.__name__}:{e}')
                explicit__all__ = None
        
        return explicit__all__, implicit__all__

def ComputeWildcards(state:State) -> Mapping[Mod, Tuple['Collection[str] | None', 'Collection[str]']]:
    """
    Maps ast Modules to the collection of names explicitely given in ``__all__`` variable.
    If ``__all__`` is not defined at the module scope, the result is None.

    Maps each modules to the list of names imported if one wildcard imports this module.
     
    A bi-product of this analysis maps each wildcard ImportFrom nodes to the collection of names they are actually importing.
    """
    
    def processModuleAST(mod_ast: ast.Module) -> Tuple['Collection[str] | None', 'Collection[str]']:
        return _ComputeWildcards(state, builder).visit_Module(mod_ast)

    builder = OrderedBuilder(state, processModuleAST, max_iterations=10)
    r = builder.build()
    print(f'done in {builder.iteration} iterations')
    return r

@attr.s()
class FullNames:
    """
    Maps definitions to their full names. 
    """
    modules:Dict[Tuple[str], ast.Module] = attr.ib()
    use_def:Dict[ast.AST, List[Def]] = attr.ib()

class CanonicalNames:
    ...

if __name__ == "__main__":
    import sys, json

    modules = sys.argv[1:]
    dump_file = modules.pop()

    proj = Project(nested_dependencies=20)

    for modname in modules:
        StateModifiers(proj.state).add_typeshed_module(modname)
    
    proj.analyze_project()
    with open(dump_file, 'w') as d:
        json.dump(proj.state.dump(), d)
