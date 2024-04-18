
from __future__ import annotations

from typing import TextIO
from dataclasses import dataclass
from io import StringIO

from unittest import TestCase

from libstatic._lib.passmanager import EventDispatcher, Event

@dataclass
class Ask(Event):
    instance: object

@dataclass
class Respond(Event):
    instance: object

class WhoAsk:
    """
    First class which ask who is listening to it
    """
    def __init__(self, event_dispatcher:EventDispatcher, fout: TextIO):
        # Save a reference to the event dispatch
        self.event_dispatcher = event_dispatcher

        # Listen for the RESPOND event type
        self.event_dispatcher.add_event_listener(Respond, self.on_answer_event)

        self.fout = fout

    def ask(self):
        """
        Dispatch the ask event
        """
        print(f"I'm instance {self}. Who are listening to me ?", file=self.fout)
        self.event_dispatcher.dispatch_event(Ask(self))

    def on_answer_event(self, event: Respond):
        """
        Event handler for the RESPOND event type
        """
        print (f"Thank you instance {event.instance}", file=self.fout)

class WhoRespond:
    """
    Second class who respond to ASK events
    """
    def __init__(self, event_dispatcher: EventDispatcher, fout: TextIO):
        # Save event dispatcher reference
        self.event_dispatcher = event_dispatcher

        # Listen for ASK event type
        self.event_dispatcher.add_event_listener(
            Ask, self.on_ask_event
        )

        self.fout = fout

    def on_ask_event(self, event: Ask):
        """
        Event handler for ASK event type
        """
        self.event_dispatcher.dispatch_event(Respond(self))

class TestEventDispatcher(TestCase):
    
    def test_dispatcher(self):
        out = StringIO()
        # Create and instance of event dispatcher
        dispatcher = EventDispatcher()

        # Create an instance of WhoAsk class and two instance of WhoRespond class
        who_ask = WhoAsk(dispatcher, out)
        
        WhoRespond(dispatcher, out); WhoRespond(dispatcher, out)

        # WhoAsk ask :-)
        who_ask.ask()

        result = out.getvalue()
        assert result.count('\n') == 3
        line1, line2, line3, empty = result.split('\n')
        assert empty == ''
        assert line1.endswith('Who are listening to me ?')
        assert line2.startswith('Thank you instance')
        assert line3.startswith('Thank you instance')
        assert line2 != line3

