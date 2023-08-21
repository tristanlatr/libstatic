"""
Provides a generic class to process a collection of objects in topological order.
"""
from collections import defaultdict
from typing import Generic, TypeVar, Optional, Iterable, Dict, List
import abc

KT = TypeVar("KT")
VT = TypeVar("VT")
RT = TypeVar("RT")

class TopologicalProcessor(abc.ABC, Generic[KT, VT, RT]):
    """
    Base class for processing objects in topological order. 
    Decoupled from the concrete types so it can be re-used 
    for several order-sensitive analysis.

    Base classes must override `getObj` and `processObj`. 
    
    The processing code can call `getProcessedObj` to indicate that the given 
    object *should* be processed before resuming the current task. In case of cycles, `getProcessedObj`
    returns a object that is not completely processed. To know wether the returned obeject is processed, one
    can use the `processing_state` mapping or the `result` mapping, a object that is still processing will not
    be present in the `result` mapping.
    """

    UNPROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.processing_state: Dict[VT, object] = defaultdict(
            lambda: TopologicalProcessor.UNPROCESSED
        )
        self.processing: List[VT] = []
        self.unprocessed: List[VT] = []
        self.result: Dict[VT, RT] = {}

    @abc.abstractmethod
    def getObj(self, key: KT) -> Optional[VT]:
        """
        Returns the object known by the given key.
        """

    @abc.abstractmethod
    def processObj(self, obj: VT) -> RT:
        """
        Process the object known by the given key and returns the 'result' of the analysis.
        If the processing code mutates a global state, there might be no need for the return value.
        """

    def getProcessedObj(self, key: KT) -> Optional[VT]:
        """
        Request processing of the given object and returns the object.
        The object might still be processing in the case of cycles.
        """
        obj = self.getObj(key)
        if obj is None:
            return None
        if self.processing_state[obj] is TopologicalProcessor.PROCESSING:
            return obj
        if self.processing_state[obj] is TopologicalProcessor.UNPROCESSED:
            self._processObj(obj)
        return obj

    def _processObj(self, obj: VT) -> None:
        assert self.processing_state[obj] is TopologicalProcessor.UNPROCESSED
        self.processing_state[obj] = TopologicalProcessor.PROCESSING
        # it can happend that obj is not in unprocessed
        # when we run fromText() several times on the same system.
        if obj in self.unprocessed:
            self.processing.append(obj)
            self.result[obj] = self.processObj(obj)
            self.unprocessed.remove(obj)
            head = self.processing.pop()
            assert head is obj

        self.processing_state[obj] = TopologicalProcessor.PROCESSED

    def process(self, objects: Iterable[VT]) -> Dict[VT, RT]:
        """
        Run the `processObj` method on all objects.
        """
        self.unprocessed = list(dict.fromkeys(objects))
        while self.unprocessed:
            obj = next(iter(self.unprocessed))
            self._processObj(obj)
        return self.result
