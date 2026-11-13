
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Protocol,
)

from .events import EventDispatcher, SupportLibraryEvent

if TYPE_CHECKING:
    from ._modules import AnyNode
    

class ISupport(Protocol):
    """
    Instances of this class carry all the required information for the passmanager to support
    concrete types of nodes like the one created by standard library L{ast} or L{astroid} or L{gast} or L{parso}.

    Currently, the only thing that needs to be known about the AST is how to iterate across
    all the children nodes. But that list might grow with the future developments

    @note: This interface is implemented by L{ast} and L{gast}.
    """

    @staticmethod
    def iter_child_nodes(node: AnyNode) -> Iterable[AnyNode]:
        """
        Callable that yields the direct child node starting at the given node inclusively. Like L{ast.iter_child_nodes}.
        """


class ASTCompat:
    """
    Wrapper to support a set of node types based on the registered strategy.
    """
    
    def __init__(self, dispatcher: EventDispatcher):
        self._support: ISupport = None # type: ignore
        dispatcher.addEventListener(SupportLibraryEvent, self._onSupportLibraryEvent)


    def _onSupportLibraryEvent(self, event: SupportLibraryEvent):
        self._support = event.lib

    def iter_child_nodes(self, node: AnyNode) -> Iterable[AnyNode]:
        return self._support.iter_child_nodes(node)

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
        todo = deque(self.iter_child_nodes(node))
        while todo:
            node = todo.popleft()
            if stopTypecheck is None or not isinstance(node, stopTypecheck):
                todo.extend(self.iter_child_nodes(node))
            if typecheck is None or isinstance(node, typecheck):
                yield node
