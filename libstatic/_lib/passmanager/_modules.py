

from __future__ import annotations

from typing import (
    Any,
    Iterator,
    Mapping,
    TYPE_CHECKING
)

import dataclasses

from .events import (EventDispatcher, ModuleAddedEvent, 
                     ModuleTransformedEvent, ModuleRemovedEvent)

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


class _Node2RootMapping(Mapping[AnyNode, ModuleNode]):
    """
    Tracks the root modules of all nodes in the system.

    Part of L{ModuleCollection}. 
    """
    def __init__(self, dispatcher: EventDispatcher, ast: ASTCompat) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleTransformedEvent, self._onModuleTransformedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        # We might be using weak keys and values dictionnary here.
        self.__data: dict[AnyNode, ModuleNode] = {}
        self._ast = ast

    def _onModuleAddedEvent(self, event: ModuleAddedEvent | ModuleTransformedEvent) -> None:
        newmod = event.mod.node
        # O(n), every time :/
        for node in self._ast.walk(newmod):
            self.__data[node] = newmod

    def _onModuleRemovedEvent(
        self, event: ModuleRemovedEvent | ModuleTransformedEvent
    ) -> None:
        # O(n), every time :/
        node = event.mod.node
        to_remove = []
        for n, r in self.__data.items():
            if r is node:
                to_remove.append(n)
        for n in to_remove:
            del self.__data[n]

    def _onModuleTransformedEvent(self, event: ModuleTransformedEvent) -> None:
        # TODO (optimizations): 2xO(n), every time: Thid could be improved by introducing 'uptdates_regions' Transformation
        # attribute that will contain a sequence of all nodes added in the tree, we also would need a sequnce
        # aof nodes removed from the tree. 
        self._onModuleRemovedEvent(event)
        self._onModuleAddedEvent(event)

    # Boring mapping interface

    def __contains__(self, __key: object) -> bool:
        return __key in self.__data

    def __getitem__(self, __key: AnyNode) -> ModuleNode:
        return self.__data[__key]

    def __iter__(self) -> Iterator[AnyNode]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)


class ModuleCollection(Mapping[str | ModuleNode | AnyNode, Module]):
    """
    A fake C{sys.modules} to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher, ast):
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[ModuleNode, Module] = {}
        self.__roots = _Node2RootMapping(dispatcher, ast)

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
                return self.__node2module[self.__roots[__key]]
            except KeyError:
                pass
        raise KeyError(__key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__name2module)

    def __len__(self) -> int:
        return len(self.__name2module)
