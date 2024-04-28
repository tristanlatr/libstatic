
from __future__ import annotations

import ast
import dataclasses
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from . import _AnalysisResult, Analysis, Module, Transformation


class Event:
    """
    Base event to use with EventDispatcher.
    """


EventListener = Callable[[Event], None]


class EventDispatcher:
    """
    Generic event dispatcher which listen and dispatch events
    """

    def __init__(self) -> None:
        self._events: dict[type[Event], list[EventListener]] = {}

    def hasListener(self, event_type: type[Event], listener: EventListener) -> bool:
        """
        Return true if listener is register to event_type
        """
        # Check for event type and for the listener
        if event_type in self._events:
            return listener in self._events[event_type]
        else:
            return False

    def dispatchEvent(self, event: Event) -> None:
        """
        Dispatch an instance of Event class
        """
        # Dispatch the event to all the associated listeners
        if type(event) in self._events:
            listeners = self._events[type(event)]

            for listener in listeners:
                listener(event)

    def addEventListener(
        self, event_type: type[Event], listener: EventListener
    ) -> None:
        """
        Add an event listener for an event type
        """
        # Add listener to the event type
        if not self.hasListener(event_type, listener):
            listeners = self._events.get(event_type, [])
            listeners.append(listener)
            self._events[event_type] = listeners

    def removeEventListener(
        self, event_type: type[Event], listener: EventListener
    ) -> None:
        """
        Remove event listener.
        """
        # Remove the listener from the event type
        if self.hasListener(event_type, listener):
            listeners = self._events[event_type]

            if len(listeners) == 1:
                # Only this listener remains so remove the key
                del self._events[event_type]

            else:
                # Update listeners chain
                listeners.remove(listener)

                self._events[event_type] = listeners


@dataclasses.dataclass(frozen=True)
class ModuleChangedEvent(Event):
    """
    When a module is transformed.
    """

    mod: Module
    """
    The module that have been transformed.
    """


@dataclasses.dataclass(frozen=True)
class ClearAnalysisEvent(Event):
    """
    When an analysis is invalidated.
    """

    analysis: type[Analysis]
    """
    The analysis type invalidated. 
    """
    
    node: ast.Module
    """
    Old the module node transformed (or added/removed).
    """


@dataclasses.dataclass(frozen=True)
class ModuleAddedEvent(Event):
    """
    When a module is added to the passmanager.
    """

    mod: Module
    """
    The module that have been added.
    """


@dataclasses.dataclass(frozen=True)
class ModuleRemovedEvent(Event):
    """
    When a module is removed from the passmanager.
    """

    mod: Module
    """
    The module that have been removed.
    """


@dataclasses.dataclass(frozen=True)
class RunningTransform(Event):
    """
    Before a transformation is run. 
    """
    
    transformation: type[Transformation]
    """
    The transformstion type.
    """

    node: ast.AST
    """
    The module node that we're transforming.
    """


@dataclasses.dataclass(frozen=True)
class TransformEnded(Event):
    """
    After a transformation has been run.
    """
    
    transformation: type[Transformation]
    """
    The transformstion type.
    """
    
    node: ast.AST
    """
    The module node that was potentially transformed.
    """


@dataclasses.dataclass(frozen=True)
class RunningAnalysis(Event):
    """
    Before an analysis is run. 
    """
    
    analysis: type[Analysis]
    """
    The analysis type.
    """
    
    node: ast.AST
    """
    The module node that we are about to transform.
    """


@dataclasses.dataclass(frozen=True)
class AnalysisEnded(Event):
    """
    After an analysis has been run.
    """
    
    analysis: type[Analysis]
    """
    The analysis type.
    """
    
    node: ast.AST
    """
    The node the analyis has ran on.
    """
    
    result: _AnalysisResult
    """
    The result of the analysis. 
    """