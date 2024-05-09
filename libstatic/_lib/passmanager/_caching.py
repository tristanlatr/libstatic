
from __future__ import annotations

from collections import defaultdict
import dataclasses
from typing import Hashable, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from typing import NoReturn
    from ._passmanager import Analysis


class AnalysisResult:
    """
    Simple wrapper for the result of an analysis.
    """
    @property
    def result(self) -> object:
        raise NotImplementedError(self.result)

    @classmethod
    def Success(self, result: object) -> AnalysisResult:
        return _AnalysisSuccess(result)

    @classmethod
    def Error(self, exception: Exception) -> AnalysisResult:
        return _AnalysisError(exception)
    
    def isError(self) -> bool:
        return isinstance(self, _AnalysisError)


@dataclasses.dataclass(frozen=True)
class _AnalysisError(AnalysisResult):
    _error: Exception

    @property
    def result(self) -> object:
        raise self._error


@dataclasses.dataclass(frozen=True)
class _AnalysisSuccess(AnalysisResult):
    _result: object

    @property
    def result(self) -> object:
        return self._result


class AnalysisCache:
    """
    The strucutre of the cache consist in nested dicts.
    But this class facilitates the messages with the module pass manager.
    """

    def __init__(self) -> None:
        self.__data: dict[type[Analysis], dict[Hashable, AnalysisResult]] = defaultdict(
            dict
        )

    def set(self, analysis: type[Analysis], node: Hashable, result: AnalysisResult):
        """
        Store the analysis result in the cache.
        """
        if analysis.do_not_cache: raise RuntimeError()
        self.__data[analysis][node] = result

    def get(self, analysis: type[Analysis], node: Hashable) -> AnalysisResult | None:
        """
        Query for the cached result of this analysis.
        """
        if analysis in self.__data:
            try:
                return self.__data[analysis][node]
            except KeyError:
                return None
        return None

    def clear(self, analysis: type[Analysis]):
        """
        Get rid of the the given analysis result.
        """
        if analysis in self.__data:
            del self.__data[analysis]

    def analysisTypes(self) -> Iterator[type[Analysis]]:
        """
        Get an iterator on all analyses types in this cache.
        """
        yield from self.__data