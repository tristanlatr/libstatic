

from __future__ import annotations

from types import SimpleNamespace
from typing import (
    Any,
    Collection,
    Container,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    TYPE_CHECKING,
    MutableMapping,
    Protocol,
    Sequence,
    TypedDict
)

import dataclasses
import ast
from weakref import WeakKeyDictionary, WeakSet

from libstatic._lib.structures import FrozenNamespace, OrderedSet

from .events import (EventDispatcher, ModuleAddedEvent, 
                     ModuleTransformedEvent, ModuleRemovedEvent)

from typing import TypeVar

import attrs

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

RootNode = Any
"""
Represent the root of a tree.
"""

AnyNode = Any
"""
Represent any kind of node in a tree as well root nodes.
"""


@dataclasses.dataclass(frozen=True)
class Module:
    """
    The specifications of a python module.
    """

    node: RootNode
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


class ITreeSupport(Protocol):
    """
    Instances of this class carries all the required information for the passmanager to support
    concrete types of trees like the one created by standard library L{ast} or L{astroid} or L{gast} or L{parso}.

    Currently, the only things that needs to be known about the tree is: 
      
      - how to iterate across the direct children of a node. 
      - get a list of identifiers a tree includes/imports/depends on.

    But that list might grow with the future developments
    """

    @staticmethod
    def children(node: AnyNode) -> Iterable[AnyNode]:
        """
        Yields the direct child node starting at the given node inclusively. Like L{ast.iter_child_nodes}.
        """
    
    @staticmethod
    def includes(node: RootNode) -> Sequence[str]:
        """
        Return a list of identifiers coresponding to the trees this one depends on.
        Typically this is the list of the imported modules. 
        Python Note: it's important to list all imports including the ones inside functions or uder TYPE_CHECKING blocks.
        """

class ASTreeSupport(ITreeSupport):
    @staticmethod
    def children(node: AnyNode) -> Iterable[AnyNode]:
        return ast.iter_child_nodes(node)
    @staticmethod
    def includes(node: RootNode) -> Sequence[str]:
        raise NotImplementedError()

@dataclasses.dataclass(frozen=True)
class TreeWalker:
    treesupport: ITreeSupport

    def walk(self, 
             node: AnyNode, 
             typecheck: type | tuple[type, ...] | None = None,
             stopTypecheck: type | tuple[type, ...] | None = None) -> Iterable[AnyNode]:
        """
        Recursively yield all nodes matching the typecheck
        in the tree starting at *node* (including *node* itself), in bfs order.

        Do not recurse on children of types matching the stopTypecheck type.
        """
        from collections import deque

        yield node
        todo = deque(self.treesupport.children(node))
        while todo:
            node = todo.popleft()
            if stopTypecheck is None or not isinstance(node, stopTypecheck):
                todo.extend(self.treesupport.children(node))
            if typecheck is None or isinstance(node, typecheck):
                yield node


# This is not considered as an analysis because it's a core part of the library
# and must be maintained before the analysis cache.
class ancestors(ast.NodeVisitor):
    """
    Associate each node with the list of its ancestors in the result attribute.
    """
    current: tuple[AnyNode, ...] | tuple[()]

    def __init__(self, astcompat: ASTCompat) -> None:
        self.result: MutableMapping[AnyNode, Sequence[AnyNode]] = {}
        """
        For each visited node, stores it's list of ancestors in this mapping.
        """
        self.current = ()
        """
        The current list of ancestors of the next node to visit.
        """

        self.__astcompat = astcompat

    def generic_visit(self, node: AnyNode) -> None:
        self.result[node] = current = self.current
        self.current += node,
        for n in self.__astcompat.iter_child_nodes(node):
            self.generic_visit(n)
        self.current = current

    visit = generic_visit

@dataclasses.dataclass(frozen=True)
class _Removal:
    "when a node is removed"
    node: AnyNode

@dataclasses.dataclass(frozen=True)
class _Addition:
    "when a node is added"
    node: AnyNode
    ancestor: AnyNode


class AncestorsMap(SupportsGetItem[AnyNode, Sequence[AnyNode]]):
    """
    Tracks the ancestors of all nodes in the system and 
    provide the special L{passmanager.ancestors} analysis.

    Part of L{ModuleCollection}. 
    """
    def __init__(self, dispatcher: EventDispatcher, astcompat: ASTCompat) -> None:
        super().__init__()
        # register the event listeners
        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleTransformedEvent, self._onModuleTransformedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)

        # Use weak keys dictionnary here.
        self.__data: WeakKeyDictionary[AnyNode, Sequence[AnyNode]] = WeakKeyDictionary()
        self.__removed: WeakSet[RootNode] = WeakSet()

        self.__astcompat = astcompat
    
    def _ancestorsWithContext(self, ancestor: AnyNode) -> ancestors:
        """
        Create a ancestors gatherer with a pre-set context coming from the given node.
        Should only be used to gather the ancestors of the direct children of the given node.
        """
        current = self.__data[ancestor]
        ans = ancestors(self.__astcompat)
        ans.current = tuple(current)
        return ans

    def _onModuleAddedEvent(self, event: ModuleAddedEvent | ModuleTransformedEvent) -> None:
        newmod = event.mod.node
        
        if newmod in self.__removed:
            self.__removed.discard(newmod)
        
        # O(Number of nodes in the module), every time, that's probably necessary
        ans = ancestors(self.__astcompat)
        ans.result = self.__data
        ans.visit(newmod)

    def _onModuleRemovedEvent(
        self, event: ModuleRemovedEvent | ModuleTransformedEvent
    ) -> None:
        # Since we use weakrefs we dont have to delete it ourselves.
        # Instead we mark the module as removed, and that's it... 
        node = event.mod.node
        self.__removed.add(node)


    def _onModuleTransformedEvent(self, event: ModuleTransformedEvent) -> None:
        t = event.transformation
        if t._updates:
            # optimizations: Avoid a O(Number of nodes in the module), every time a mode is transformed
            for u in t._updates:
                # It's O(Number of nodes added+removed)
                if isinstance(u, _Addition):
                    ans = self._ancestorsWithContext(u.ancestor)
                    ans.result = self.__data
                    ans.visit(u.node)

                elif isinstance(u, _Removal):
                    for n in self.__astcompat.walk(u.node):
                        if n in self.__data:
                            del self.__data[n]
                else:
                    raise TypeError(f'unexpected update type: {u}')
        else:
            # Not optimized
            self._onModuleRemovedEvent(event)
            self._onModuleAddedEvent(event)

    def _hasBeenRemoved(self, node: object) -> bool:
        """
        Since we use weak key mapping, we don't manage the deletion of values ourselve.
        We trust the weak key mapping to do the job, but if we still have another reference to the object
        we must maintain a set of removed modules.
        """
        return node in self.__removed or bool((
            ans := self.__data[node] # if that raises, the node is not in the system :/
            ) and ans[0] in self.__removed)

    # mapping-ish interface

    def __contains__(self, __key: object) -> bool:
        return (__key in self.__data 
                and not self._hasBeenRemoved(__key))

    def __getitem__(self, __key: AnyNode) -> RootNode:
        if self._hasBeenRemoved(__key):
            raise KeyError(__key) # module has been removed
        return self.__data[__key]

    def get(self, key: AnyNode, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def _data(self) -> WeakKeyDictionary[AnyNode, Sequence[AnyNode]]:
        return self.__data

    def _merge(self, other: AncestorsMap) -> None:
        self.__data.update(other._data())

    def __iter__(self) -> Iterator[AnyNode]:
        raise NotImplementedError('this "mapping" is not iterable')

    def __len__(self) -> int:
        raise NotImplementedError('this "mapping" is not sized')


# TODO: rename me for Forest
class ModuleCollection(Mapping['str | RootNode | AnyNode', Module]):
    """
    A smart mapping to contain the pass manager modules.

    To be used like a read-only mapping where the values can be accessed
    both by module name or by module ast node (alternatively by any node contained in a known module).
    """

    def __init__(self, dispatcher: EventDispatcher, astcompat: ASTCompat) -> None:
        self.__name2module: dict[str, Module] = {}
        self.__node2module: dict[RootNode, Module] = {}
        
        self.ancestors = AncestorsMap(dispatcher, astcompat); "The ancestors"

        dispatcher.addEventListener(ModuleAddedEvent, self._onModuleAddedEvent)
        dispatcher.addEventListener(ModuleRemovedEvent, self._onModuleRemovedEvent)
    
    def _merge(self, other: ModuleCollection) -> None:
        for mod in other.values():
            self._add(mod)
        self.ancestors._merge(other.ancestors)
    
    def _add(self, module: Module) -> None:
        modname = module.modname
        modnode = module.node

        if self.get(modname) not in (None, module):
            raise ValueError(
                f"a module named {modname!r} " f"already exist: {self[modname]}"
            )

        if self.get(modnode) not in (None, module):
            raise ValueError(
                f"the ast of the module {modname!r} is already "
                f"associated with another module: {self[modnode]}"
            )

        # register the module as beeing a part of this collection.
        self.__name2module[modname] = module
        self.__node2module[modnode] = module

    def _onModuleAddedEvent(self, event: ModuleAddedEvent) -> None:
        self._add(event.mod)

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

    def __getitem__(self, __key: str | RootNode | AnyNode) -> Module:
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


RootNodeT = TypeVar('RootNodeT')
AttributesT = TypeVar('AttributesT')

class Tree(Generic[RootNodeT, AttributesT]):
    """
    Encapsulate a single tree. This is a read-only datastructure.
    
    All roots are required to have an identifier.  Typically this is the module name.
    """
    def __init__(self, root: RootNodeT, identifier: str, **attributes: Hashable) -> None:
        self.__root = root
        self.__identifier = identifier
        
        self.attributes: AttributesT = FrozenNamespace(**attributes)
    
    @property
    def root(self) -> RootNodeT:
        return self.__root
    
    @property
    def identifier(self) -> str:
        return self.__identifier

    def __hash__(self) -> int:
        return hash((self.root, self.identifier, self.attributes))
    
    def __eq__(self, other: object) -> bool:
        if isinstance(self, Tree) and isinstance(other, Tree):
            return self.root == other.root and \
                self.identifier == other.identifier and \
                self.attributes == other.attributes
        return NotImplemented

class Forest(Collection[Tree[RootNodeT, AttributesT]]):
    """
    A collection of trees. 
    """
    def __init__(self, trees: Iterable[Tree]=None) -> None:

        # each operation must maintain these 3 structures.
        self.__identifier2tree: dict[str, Tree[RootNodeT, AttributesT]] = {}
        self.__root2tree: dict[RootNode, Tree[RootNodeT, AttributesT]] = {}
        self.__trees: OrderedSet[Tree[RootNodeT, AttributesT]] = OrderedSet()

        if trees is not None:
            for t in trees:
                self.add(t)
    
    def add(self, tree: Tree[RootNodeT, AttributesT]) -> None:

        if tree in self:
            return

        if self.get(tree.identifier):
            raise ValueError(
                f"identifier {tree.identifier!r} " 
                f"if already taken: {self[tree.identifier]}"
            )

        if self.get(tree.root):
            raise ValueError(
                f"root node {tree.identifier!r} is already "
                f"associated with another tree: {self[tree.root]}"
            )

        # add the tree in the collection.
        self.__identifier2tree[tree.identifier] = tree
        self.__root2tree[tree.root] = tree
        self.__trees.add(tree)
    
    def remove(self, tree: Tree[RootNodeT, AttributesT]) -> None:
        if tree in self:
            raise ValueError(f"tree not in the collection: {tree}")

        # remove the tree from the collection
        del self.__identifier2tree[tree.identifier]
        del self.__root2tree[tree.root]
        self.__trees.discard(tree)
    
    #  getitem interface

    def __getitem__(self, __key: str | RootNodeT) -> Tree[RootNodeT, AttributesT]:
        if isinstance(__key, str):
            return self.__identifier2tree[__key]
        else:
            return self.__root2tree[__key]
    
    def get(self, key: str | RootNodeT, default:Any=None) -> Tree[RootNodeT, AttributesT] | None:
        try:
            return self[key]
        except KeyError:
            return default
    
    # collection interface
    
    def __iter__(self) -> Iterator[Tree[RootNodeT, AttributesT]]:
        return iter(self.__trees)

    def __len__(self) -> int:
        return len(self.__trees)

    def __contains__(self, other: object) -> bool:
        # A forest contains trees, root nodes and identifiers.
        return other in self.__trees or \
            other in self.__identifier2tree or \
            other in self.__root2tree

if TYPE_CHECKING:
    class ASTAttributes:
        'only for typing'
        
        filename: str | None 
        """
        The filename of the source file.
        """

        isPackage: bool
        """
        Whether the module is a package.
        """
        
        # TODO: namespace packages are not supported at the moment.
        # is_namespace_package: bool
        # """
        # Whether the module is a namespace package.
        # """
        
        isStub: bool
        """
        Whether the module is a stub module.
        """
        
        sourceCode: str | None
        """
        The source.
        """
    AbstractSyntaxTree = Tree[ast.Module, ASTAttributes]
    ASTForest = Forest[ast.Module, ASTAttributes]
else:
    AbstractSyntaxTree = Tree
    ASTForest = Forest