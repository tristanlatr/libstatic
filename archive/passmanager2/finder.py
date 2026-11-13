"""
Create L{Module} instances. 
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator

from . import Module

@dataclasses.dataclass(frozen=True)
class SearchContext:
    """
    A fake sys.path to search for modules. See typeshed.
    """

class Finder:
    """
    In charge of finding python modules and creating L{Module} instances for them.
    """

    search_context: SearchContext

    def module_by_name(self, modname: str) -> Module:
        """
        Find a module by name based on the current search context.
        """
        raise NotImplementedError()

    def modules_by_path(self, path: Path) -> Iterator[Module]:
        """
        If the path points to a directory it will yield recursively
        all modules under the directory. Pass a file and it will alway yield one Module entry.
        """
        raise NotImplementedError()