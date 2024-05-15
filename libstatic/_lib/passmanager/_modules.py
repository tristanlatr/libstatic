

from __future__ import annotations

from typing import (
    Any,
    Iterator,
    Mapping,
    TYPE_CHECKING,
    Protocol
)

import dataclasses
import ast
from weakref import WeakKeyDictionary, WeakSet

from .events import (EventDispatcher, ModuleAddedEvent, 
                     ModuleTransformedEvent, ModuleRemovedEvent)

from typing import TypeVar

_KT_contra = TypeVar("_KT_contra", contravariant=True)
_VT_co = TypeVar("_VT_co", covariant=True)
class SupportsGetItem(Protocol[_KT_contra, _VT_co]):
    def __contains__(self, x: Any, /) -> bool: ...
    def __getitem__(self, key: _KT_contra, /) -> _VT_co: ...
    def get(self, item, default=None):
        ...

if TYPE_CHECKING:
    from ._astcompat import ASTCompat

__docformat__ = 'epytext'

ModuleNode = Any
"""
Symbol that represent a ast module node. 
"""

AnyNode = Any
"""
Symbol that represent any kind of ast node. 
"""


@dataclasses.dataclass(frozen=True)
class Module:
    """
    The specifications of a python module.
    """

    node: ModuleNode
    """
    The module node.
    """
    
    modname: str
    """
    The module fully qualified name. 
    If the module is a package, do not include C{__init__}
    """
    
    filename: str | None = None
    """
    The filename of the source file.
    """
    
    is_package: bool = False
    """
    Whether the module is a package.
    """
    
    # TODO: namespace packages are not supported at the moment.
    # is_namespace_package: bool = False
    # """
    # Whether the module is a namespace package.
    # """
    
    is_stub: bool = False
    """
    Whether the module is a stub module.
    """
    
    code: str | None = None
    """
    The source.
    """

def _getAncestors(astcompat: ASTCompat):
    class ancestors(ast.NodeVisitor):
        '''
        Associate each node with the list of its ancestors in the result attribute.
        '''

        current: tuple[AnyNode, ...] | tuple[()]

        def __init__(self):
            self.result: dict[AnyNode, list[AnyNode]] = {}
            self.current = ()

        def generic_visit(self, node):
            self.result[node] = current = self.current
            self.current += node,
            for n in astcompat.iter_child_nodes(node):
                self.generic_visit(n)
            self.current = current

        visit = generic_visit

        # def doPass(self, node: ast.Module) -> dict[ast.AST, list[ast.AST]]:
        #     self.visit(node)
        #     return self.result

    return ancestors

if TYPE_CHECKING:
    class ancestors:
        result: dict[AnyNode, list[AnyNode]]
        current: tuple[AnyNode, ...]
        def visit(self, node):...

@dataclasses.dataclass(frozen=True)
class _Removal:
    "when a node is removed"
    node: AnyNode

@dataclasses.dataclass(frozen=True)
class _Addition:
    "when a node is added"
    node: AnyNode
    ancestor: AnyNode


class AncestorsMap(SupportsGetItem[AnyNode, list[AnyNode]]):
    """
    Tracks the ancestors of all nodes in the system and 
    provide the special L{passmanager.ancestors} analysis.

    Part of L{ModuleCollection}. 
    """
    def __init__(self, dispatcher: EventDispatcher, ast: ASTCompat) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleTransformedEvent, self._onModuleTransformedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        # Use weak keys dictionnary here.
        self.__data: dict[AnyNode, list[AnyNode]] = WeakKeyDictionary()
        self.__removed: set[ModuleNode] = WeakSet()

        self._ast = ast
        self._ancestorsType = _getAncestors(ast)
    
    def ancestorsWithContext(self, ancestor: AnyNode) -> ancestors:
        current = self.__data[ancestor]
        ans = self._ancestorsType()
        ans.current = tuple(current)
        return ans

    def _onModuleAddedEvent(self, event: ModuleAddedEvent | ModuleTransformedEvent) -> None:
        newmod = event.mod.node
        
        if newmod in self.__removed:
            self.__removed.discard(newmod)
        
        # O(Number of nodes in the module), every time, that's probably necessary
        ans = self._ancestorsType()
        ans.result = self.__data
        ans.visit(newmod)

    def _onModuleRemovedEvent(
        self, event: ModuleRemovedEvent | ModuleTransformedEvent
    ) -> None:
        # Since we use weakrefs we dont have to delete the ourselves.
        # Instead we mark the module as removed, and that's it... 
        node = event.mod.node
        self.__removed.add(node)

        
    def _onModuleTransformedEvent(self, event: ModuleTransformedEvent) -> None:
        t = event.transformation
        if t._updates is not None:
            # optimizations: Avoid a O(Number of nodes in the module), every time a mode is transformed
            for u in t._updates:
                # It's O(Number of nodes added+removed)
                if isinstance(u, _Addition):
                    ans = self.ancestorsWithContext(u.ancestor)
                    ans.result = self.__data
                    ans.visit(u.node)

                elif isinstance(u, _Removal):
                    for n in self._ast.walk(u.node):
                        if n in self.__data:
                            del self.__data[n]
                else:
                    raise TypeError(f'unexpected update type: {u}')
        else:
            # Not optimized
            self._onModuleRemovedEvent(event)
            self._onModuleAddedEvent(event)

    def _hasBeenRemoved(self, node) -> bool:
        return node in self.__removed or (
            ans := self.__data[node] # if that raises, the node is not in the system :/
            ) and ans[0] in self.__removed
    
    # mapping-ish interface

    def __contains__(self, __key: object) -> bool:
        return (__key in self.__data 
                and not self._hasBeenRemoved(__key))

    def __getitem__(self, __key: AnyNode) -> ModuleNode:
        if self._hasBeenRemoved(__key):
            raise KeyError(__key) # module has been removed
        return self.__data[__key]

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default
        
    def __iter__(self) -> Iterator[AnyNode]:
        raise NotImplementedError('this "mapping" is not iterable')

    def __len__(self) -> int:
        raise NotImplementedError('this "mapping" is not sized')


class ModuleCollection(Mapping[str | ModuleNode | AnyNode, Module]):
    """
    A smart mapping to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher, ast):
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[ModuleNode, Module] = {}
        
        self.ancestors = AncestorsMap(dispatcher, ast); "The ancestors"

        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if self.get(modname) not in (None, mod):
            raise ValueError(
                f"a module named {modname!r} " f"already exist: {self[modname]}"
            )

        if self.get(modnode) not in (None, mod):
            raise ValueError(
                f"the ast of the module {modname!r} is already "
                f"associated with another module: {self[modnode]}"
            )

        # register the module as beeing a part of this collection.
        self.__name2module[modname] = mod
        self.__node2module[modnode] = mod

    def _onModuleRemovedEvent(self, event: ModuleRemovedEvent) -> None:
        mod = event.mod
        modname = mod.modname
        modnode = mod.node

        if modname not in self or modnode not in self:
            raise ValueError(f"looks like this module is not in the collection: {mod}")

        # remove the module from the collection
        del self.__name2module[modname]
        del self.__node2module[modnode]

    # Mapping interface

    def __getitem__(self, __key: str | ModuleNode | AnyNode) -> Module:
        if isinstance(__key, str):
            return self.__name2module[__key]
        try:
            return self.__node2module[__key]
        except KeyError:
            try: 
                return self.__node2module[self.ancestors[__key][0]]
            except (KeyError, IndexError):
                pass
        raise KeyError(__key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__name2module)

    def __len__(self) -> int:
        return len(self.__name2module)
