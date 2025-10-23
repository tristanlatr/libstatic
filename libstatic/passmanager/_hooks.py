"""
Implements the hooks and plugin architecture.
"""

from __future__ import annotations

import abc
from collections import deque
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from inspect import signature, Parameter
from itertools import chain, product
import weakref

from typing import (
    Callable,
    Collection,
    Container,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from typing import NotRequired, TypeAlias, TypedDict
else:
    final = lambda f: f
    TypedDict = object

from libstatic._lib.structures import (
    Cache,
    Indexer,
    FrozenDict,
    OrderedSet,
)

import attrs



# THookObj = TypeVar('THookObj', 'UnpreparedPass', 'PreparedPass', 'CompletedPass', 'FailedPass')
# Hook: 'TypeAlias' = Callable[[THookObj], THookObj | None]
# """
# A hook is callable that is used to customize the logic 
# just before or after a pass runs. 
# Note that the hooks won't be called when retreiving 
# results from the cache; only when a pass actually runs.

# If the hook function returns a value, it'a assumed to replace 
# the given object in parameter; this should be of the same type.

# If the hook runs at the 'unprepared' step, the object type will be an L{UnpreparedPass}
# instance, at 'prepared' the object type will be L{PreparedPass} instance, and,
# at 'completed' the object type will be a L{CompletedPass} isntance.

# You cannot mutate the object in-place since it's a frozen class. But you can replace
# the given object by returning a non-None value.

# Use the `Hooks.install()` method to add a new hook. The supported
# parameters are: 
#     - hook: Hook - the callable
#     - when: 'unprepared' or 'prepared' or 'completed'
#     - kind: 'analysis' or 'transformation'
#     - level:  'node', 'tree' or 'forest'
#     - knowledge: 'node',  'tree',  'forest' - only for 'completed' hooks.
# """

# class IPluginRegistrar(Protocol):
#     hooks: Hooks
#     def configure(passe: PassLike, name: str) -> None: ...

# class IPluginFactory(Protocol):
#     def __call__(self) -> IPlugin: ...

# class IPlugin(Protocol):
#     """
#     An plugin class:
#         - has a name
#         - can support a variety of different parsers, use 'all' special word to indicate 
#             a plugin supports all kind of parsers.
#         - can handle gather()/apply()/run() keywords, aka "run options"
#         - can handle analysis()/transformation() keywords, aka "pass options"
#         - can handle passe yield points keywords, aka "completed passe's attributes"
#         - can installs hooks
#         - can configure a pass alias as string
#         - register method MUST at least return self, 
#             - Returned instance of the plugin will be stored in the PassManager locals 
#               as a attribute of the given `name`. 
#             - A plugin can register other plugins and yield their instances
#               from the register() method such that they will be stored 
#               in the PassManager locals as well.
        
#     The PassManager only handled zero-argument callabled returning an instance of IPlugin.
#     """
#     name: str
#     def register(self, r: IPluginRegistrar) -> Iterable[IPlugin]:...


# class Trigger(IntEnum):
#     UNPREPARED = 24
#     PREPARED = 23
#     COMPLETED = 22
#     FAILED = 21


# class _HookedAll(IntEnum):
#     # for Hooks.install() method
#     ALL = 99
# class _HookedNotApplicable(IntEnum):
#     # for Hooks.install() method
#     NA = 0


# def _upper_if_string(v: object):
#     if isinstance(v, str):
#         return v.upper()
#     return v

# @attrs.frozen(slots=True)
# class Hooks:
#     """
#     Container for the customization hooks. 
#     The hooks are a manner to customize the process of running a pass.
    
#     >>> my_cb = lambda o: print(o)
#     >>> h = Hooks()
#     >>> h.install(my_cb, when='after', kind='analysis', 
#     ... level='tree', knowledge='all')
#     >>> cbs = h.get(_AFTER, _ANALYSIS, _TREE, _TREE)
#     >>> len(cbs)
#     1
#     >>> cbs[0](2)
#     2
#     """

#     _hooks: Indexer[tuple[Hook, int, int, int, int]] = attrs.field(default_factory=lambda: Indexer(
#         ['hook', 'when', 'kind', 'level', 'knowledge']))

#     @staticmethod
#     def _cast_string_values(when: str | int, 
#                 kind: str | int, 
#                 level: str | int, 
#                 knowledge: str | int) -> tuple[int, int, int, int]:
#         # Cast everything to instances of integers.

#         def _cast(v: str | int, maps: Iterable[type[IntEnum]]) -> int:
#             if isinstance(v, int):
#                 return v
#             for m in maps:
#                 try: return m[v]
#                 except KeyError: continue
#             raise KeyError(v)
        
#         return (
#             _cast(when, [Trigger]), 
#             _cast(kind, [PassKind, _HookedAll]),
#             _cast(level, [Level, _HookedAll]), 
#             _cast(knowledge, [_HookedNotApplicable, Level, _HookedAll])
#         )
        
#     def install(self, hook: Hook, *, 
#         when: str | Trigger,
#         kind: str | PassKind | _HookedAll,
#         level: str | Level | _HookedAll,
#         knowledge: str | Level | _HookedAll | _HookedNotApplicable = _HookedNotApplicable.NA, 
#                 ) -> None:  
#         """
#         Add a hook to the system.
#         """
#         # TODO: Use a priority-based order to apply hooks like we do for pydoctor's post-processing.
#         # indtroduce the parameter priority. 

#         when, kind, level, knowledge = self._cast_string_values(
#             *map(_upper_if_string, [when, kind, level, knowledge]))

#         # NA must always be used when the hooks runs before the pass, so validate that
#         # the only trigger that run after the pass run is the COMPLETED.
#         # TODO: Write nice error messages.
#         if when != Trigger.COMPLETED: 
#             if knowledge != _HookedNotApplicable.NA: 
#                 raise TypeError
#         elif knowledge == _HookedNotApplicable.NA: 
#             raise TypeError

#         # Some combinaison of level/knowledge makes no sens: 
#         # when the 'knowledge' is lower than the 'level'.
#         if knowledge < level: 
#             raise ValueError

#         kinds = list(PassKind) if kind == _HookedAll.ALL else [kind]
#         levels = list(Level) if level == _HookedAll.ALL else [level]
#         knowledges = list(Level) if knowledge == _HookedAll.ALL else [knowledge]
        
#         for combo in product(kinds, levels, knowledges):
#             self._hooks.add((hook, when, *combo))

#     def get(self, 
#         when: Trigger,
#         kind: PassKind,
#         level: Level,
#         knowledge: Level | _HookedNotApplicable.NA = _HookedNotApplicable.NA, 
#                 ) -> Iterable[Hook]:
#         # This method doesn't support passing string values like install() for performance reason.
#         return (k[0] for k in self._hooks.search(when=when, 
#                                                  kind=kind, 
#                                                  level=level, 
#                                                  knowledge=knowledge))

#     def uninstall(self, hook: Hook) -> None:
#         hooks = self._hooks
#         for k in hooks.search(hook=hook):
#             hooks.discard(k)
