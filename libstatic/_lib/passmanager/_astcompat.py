
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Protocol,
)

from beniget.ordered_set import ordered_set # type: ignore

from .events import EventDispatcher, SupportLibraryEvent

if TYPE_CHECKING:
    from ._modules import AnyNode
    

class ILibrarySupport(Protocol):
    """
    Instances of this class carry all the required information for the passmanager to support
    concrete types of nodes like the one created by standard library L{ast} or L{astroid} or L{gast} or L{parso}.

    Currently, the only thing that needs to be known about the AST is how to iterate across
    all the children nodes. But that list might grow with the future developments
    """

    @staticmethod
    def iter_child_nodes(node: AnyNode) -> Iterable[AnyNode]:
        """
        Callable that yields the direct child node starting at the given node inclusively. Like L{ast.iter_child_nodes}.
        If the given node is not one of the supported types, the function must raise L{NotImplementedError}.
        """


class ASTCompat:
    """
    Wrapper to support multiple concrete types of nodes based on registered strategies.
    """
    
    def __init__(self, dispatcher: EventDispatcher):
        self._supports: ordered_set[ILibrarySupport] = ordered_set()
        dispatcher.addEventListener(SupportLibraryEvent, self._onSupportLibraryEvent)


    def _onSupportLibraryEvent(self, event: SupportLibraryEvent):
        self._supports.add(event.lib)


    def iter_child_nodes(self, node: AnyNode) -> Iterable[AnyNode]:
        """
        Like L{ast.iter_child_nodes}.
        """
        for lib in self._supports:
            try:
                it = lib.iter_child_nodes(node)
            except NotImplementedError:
                continue
            else:
                return it
        raise TypeError(f'node type not supported: {node}')
    

    def walk(self, node: AnyNode, typecheck: type | None = None, stopTypecheck: type | None = None):
        """
        Recursively yield all nodes matching the typecheck
        in the tree starting at *node* (B{excluding} *node* itself), in bfs order.

        Do not recurse on children of types matching the stopTypecheck type.
        """
        from collections import deque

        todo = deque(self.iter_child_nodes(node))
        while todo:
            node = todo.popleft()
            if stopTypecheck is None or not isinstance(node, stopTypecheck):
                todo.extend(self.iter_child_nodes(node))
            if typecheck is None or isinstance(node, typecheck):
                yield node
