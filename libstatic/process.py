"""
Provides a generic class to process a collection of modules.
"""
from collections import defaultdict
from typing import Generic, TypeVar, Optional, Iterable, Dict, List
import enum
import abc

ModType = TypeVar("ModType")
BuildType = TypeVar("BuildType")


class _ProcessingState(enum.Enum):
    UNPROCESSED = 0
    PROCESSING = 1
    PROCESSED = 2


class Processor(abc.ABC, Generic[ModType, BuildType]):
    """
    Base class for processing modules in order. Decoupled from the concrete types
    so it can be re-used for several inter-modules analysis.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.state: Dict[ModType, _ProcessingState] = defaultdict(
            lambda: _ProcessingState.UNPROCESSED
        )
        self.processing_modules: List[ModType] = []
        self.unprocessed_modules: List[ModType] = []

        self.result: Dict[ModType, BuildType] = {}

    @abc.abstractmethod
    def getModule(self, name: str) -> Optional[ModType]:
        ...

    @abc.abstractmethod
    def processModule(self, mod: ModType) -> BuildType:
        ...

    def getProcessedModule(self, modname: str) -> Optional[ModType]:
        # might return a processing module in the case of cyclic imports
        mod = self.getModule(modname)
        if mod is None:
            return None
        if self.state[mod] is _ProcessingState.PROCESSING:
            return mod
        if self.state[mod] is _ProcessingState.UNPROCESSED:
            self._processModule(mod)
        return mod

    def _processModule(self, mod: ModType) -> None:
        assert self.state[mod] is _ProcessingState.UNPROCESSED
        self.state[mod] = _ProcessingState.PROCESSING
        # it can happend that mod is not in unprocessed_modules
        # when we run fromText() several times on the same system.
        if mod in self.unprocessed_modules:
            self.processing_modules.append(mod)
            self.result[mod] = self.processModule(mod)
            self.unprocessed_modules.remove(mod)
            head = self.processing_modules.pop()
            assert head is mod

        self.state[mod] = _ProcessingState.PROCESSED

    def process(self, modules: Iterable[ModType]) -> Dict[ModType, BuildType]:
        self.unprocessed_modules = list(dict.fromkeys(modules))
        while self.unprocessed_modules:
            mod = next(iter(self.unprocessed_modules))
            self._processModule(mod)
        return self.result
