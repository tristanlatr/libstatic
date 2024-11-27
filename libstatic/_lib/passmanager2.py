from __future__ import annotations
from abc import abstractmethod
from enum import IntEnum
from collections import defaultdict
from functools import reduce
import operator
from typing import Any, ClassVar, Collection, Generic, Iterable, Sequence, TypeVar, TypeAlias, Hashable

from libstatic._lib.structures import FrozenNamespace, Cache, CallResult, OrderedSet

########## Caching configurations

# 2 #: highest level of knowledge, it can use anything from any other modules
# 1 #: normal level of knowledge, it can use anything is the current module
# 0 #: lowest level of knowledge it can only look at the provided node

_CacheKeyLabels: frozenset[str] = OrderedSet(
    (
     'analysis', # string that represents the analysis
     # analysis modifiers
     #  'runsOn', # integer
     'knowledge', # integer
     'completeness',  # boolean
     'args' # frozen namespace

     'module', # string or None
     'node', # hashable or None
     ))

# This serves more as internal documentation rather typing
_CacheKeyT: TypeAlias = (
    tuple[str, int, bool, FrozenNamespace, str | None, Hashable | None])

########## Cache factory

def mkcache() -> Cache[_CacheKeyT, CallResult]:
    """
    
    """
    cache: Cache[_CacheKeyT, CallResult] = Cache(
        keys=_CacheKeyLabels, skipKeys=('node',))
    return cache

########## Pass **declaration** interfaces

FOREST = 2 #: global system level
MODULE = 1 #: module (MTree) level
NODE = 0 #: node level


class IPass:

    # a pass should be a class declaration that have at least a doPass() method
    # and a runsOn class attribute

    runsOn: ClassVar[int]
    
    @abstractmethod
    def doPass(cls, node: Any) -> Any:
        ...

class IPassWithDeps(IPass):

    dependencies: ClassVar[tuple[type, ...]]

class IPassWithDynamicDeps(IPass):

    @classmethod
    def dependencies(cls, args: FrozenNamespace) -> tuple[type, ...]:
        ...

class IPassWithRequiredParams(IPass):
    
    requiredParameters: ClassVar

class IPassWithOptionalParams(IPass):
    
    optionalParameters: ClassVar

T = TypeVar('T')

class IAnalysis(IPass, Generic[T]):
    
    shouldBeCached: ClassVar[bool] = True
    
    def doPass(cls, node: Any) -> T:
        ...

class IAnalysisProvidingMutator(IAnalysis):
    
    @classmethod
    def getMutator(cls):
        ...

class IAnalysisProvidingGenericMutator(IAnalysis):
    
    @classmethod
    def getGenericMutator(cls):
        ...

PreservedAnalyses: TypeAlias = type

class ITransformation(IPass):
    
    def doPass(cls, node: Any) -> Iterable[PreservedAnalyses]:
        ...

def analysis_like(analysis: IPassWithRequiredParams | IPassWithOptionalParams, 
                 **kwargs: Hashable) -> object:...

# function decorators to create analyses
# @analysis(on=FOREST/MODULE/NODE, 
#   cached=True/False, file_cached=False/True, serializer=dict(loads=json.loads, dumps=json.dumps)
#   params=('arg1',), 
#   optional_params=dict(arg2='default'), 
#   name=lambda args: args.thing.__name__, # the name must be an identifier
#   dependencies=lambda args: [mro] if args.inherited else [] / tuple[],
#   completeness=lambda result: bool(result.warnings),
#   # TODO: how can forest mutators depend on module analyses?
#   mutators=dict(add_node=lambda result, new_node, parent_node:..., # only for module or forest analyses.
#                 remove_node=lambda result, old_node:..., # only for module or forest analyses.
#                 add_module=lambda passmanager, result, new_tree:..., # only for forest analyses.
#                 remove_module=lambda passmanager, result, old_tree:..., # only for forest analyses.
#       ), 
#   ) 
# 
# def class_graph(ns: Namespace, run_on: Forest) -> ClsGraph:
#   ns.analysis_errors = []
#   ...

# @class_graph.set_generic_mutator.add_mtree()
# def class_graph_add_module(ns: Namespace, result: ClsGraph, new_module: Module)
# 
# function decorators to create transformations
# @transformation(on=MODULE, 
#   params=(), 
#   optional_params={}, 
#   dependencies=lambda args: [mro] if args.inherited else [],
#   )
# def expand_imports(ns: Namespace, node: ast.Module) -> list:
#   ...
#   ns contains
#   ns does not contains a reference to the PassManager
#   ns.ctx: a PassContext
#   ns.deps: a namespace with the lazy bounded dependency 
#   ns.args
#   ns.mutators

def transformation(type) -> type:
    ...

def analysis(type) -> type:
    ...