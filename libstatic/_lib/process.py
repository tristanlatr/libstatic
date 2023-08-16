"""
Provides a generic class to process a collection of modules.
"""
from collections import defaultdict
from typing import Generic, TypeVar, Optional, Iterable, Dict, List
import abc

ModType = TypeVar("ModType")
BuildType = TypeVar("BuildType")

class TopologicalProcessor(abc.ABC, Generic[ModType, BuildType]):
    """
    Base class for processing modules in topological order. 
    Decoupled from the concrete types so it can be re-used for several inter-modules analysis.

    Base classes must override `getModule` and `processModule`. 
    
    The processing code can call `getProcessedModule` in order to indicate that the given 
    module *should* be processed before resuming the current task. In case of cycles, `getProcessedModule`
    returns a module that is not completely processed. To know wether the returned module is processed, one
    can use the `processing_state` mapping or the `result` mapping, a module that is still processing will not
    be present in the `result` mapping.
    """

    UNPROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.processing_state: Dict[ModType, object] = defaultdict(
            lambda: TopologicalProcessor.UNPROCESSED
        )
        self.processing_modules: List[ModType] = []
        self.unprocessed_modules: List[ModType] = []
        self.result: Dict[ModType, BuildType] = {}

    @abc.abstractmethod
    def getModule(self, name: str) -> Optional[ModType]:
        """
        Returns the module with the given name.
        """

    @abc.abstractmethod
    def processModule(self, mod: ModType) -> BuildType:
        """
        Process this module and returns the 'result' of the analysis.
        If the processing code mutates the state, there is no need for the return value.
        """

    def getProcessedModule(self, modname: str) -> Optional[ModType]:
        """
        Request processing of the given module.
        """
        # might return a processing module in the case of cyclic imports
        mod = self.getModule(modname)
        if mod is None:
            return None
        if self.processing_state[mod] is TopologicalProcessor.PROCESSING:
            return mod
        if self.processing_state[mod] is TopologicalProcessor.UNPROCESSED:
            self._processModule(mod)
        return mod

    def _processModule(self, mod: ModType) -> None:
        assert self.processing_state[mod] is TopologicalProcessor.UNPROCESSED
        self.processing_state[mod] = TopologicalProcessor.PROCESSING
        # it can happend that mod is not in unprocessed_modules
        # when we run fromText() several times on the same system.
        if mod in self.unprocessed_modules:
            self.processing_modules.append(mod)
            self.result[mod] = self.processModule(mod)
            self.unprocessed_modules.remove(mod)
            head = self.processing_modules.pop()
            assert head is mod

        self.processing_state[mod] = TopologicalProcessor.PROCESSED

    def process(self, modules: Iterable[ModType]) -> Dict[ModType, BuildType]:
        """
        Run the `processModule` method on all modules.
        """
        self.unprocessed_modules = list(dict.fromkeys(modules))
        while self.unprocessed_modules:
            mod = next(iter(self.unprocessed_modules))
            self._processModule(mod)
        return self.result