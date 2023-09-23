"""
This module provides interfaces for modular low-level analyses. 
Making it easy to replace the implemetation of an analysis by another one, keeping
the same code eveywhere else.
"""

from __future__ import annotations

from typing import Collection, Mapping, Sequence, Tuple, TypeVar, Generic, Any, Type, TYPE_CHECKING
import ast
from inspect import _ParameterKind as ParameterKind
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from typing import Protocol, TypeAlias
else:
    Protocol = object

Tnode = TypeVar('Tnode')
Tresult = TypeVar('Tresult')

class IDef(Protocol):
    """
    A definition.
    """
    @property
    def node(self) -> ast.AST:...
    @property
    def islive(self) -> bool:...
    def name(self) -> str | None:...
    def add_user(self, node: IDef) -> None:...
    def users(self) -> Collection[IDef]:...

class IModuleSpec(Protocol):
    """
    A module specification.
    """
    @property
    def node(self) -> ast.Module:...
    @property
    def modname(self) -> str:...
    @property
    def filename(self) -> str:...
    @property
    def is_package(self) -> bool:...

class IBenigetAnalysis(Protocol):
    """
    The result of bgeniget analyses. Since beniget provides several analysis in a single
    visitor run, we gather all results into a single class.
    """
    @property
    def chains(self) -> IDefUseChains:...
    @property
    def locals(self) -> ILocals:...

class IImport(Protocol):
    """
    An import binding.
    """
    @property
    def orgmodule(self) -> str:...
    @property
    def orgname(self) -> str | None:...

class IMinimalProjectDefs(Protocol):
    @property
    def chains(self) -> IDefUseChains:...
    @property
    def use_chains(self) -> IUseDefChains:...
    @property
    def modules(self) -> IModuleCollection:...
    @property
    def imports(self) -> IResolvedImports:...
    @property
    def ancestors(self) -> IAncestors:...

class IProjectDefs(IMinimalProjectDefs, Protocol):
    @property
    def unreachable(self) -> IUnreachable:...
    @property
    def mros(self) -> IMROs:...
    @property
    def dunder_all(self) -> IDunderAll:...

class IArgument(Protocol):
    @property
    def node(self) -> ast.arg:...
    @property
    def kind(self) -> ParameterKind:...
    @property
    def default(self) -> ast.expr | None:...

IAncestors: TypeAlias = Mapping[ast.AST, Sequence[ast.AST]]
IDefUseChains: TypeAlias = Mapping[ast.AST, IDef]
ILocals: TypeAlias = Mapping[ast.AST, Sequence[IDef]]
IUseDefChains: TypeAlias = Mapping[ast.AST, Sequence[IDef]]
IUnreachable: TypeAlias = Collection[ast.AST]
IInstanceVars: TypeAlias = ILocals
IResolvedImports: TypeAlias = Mapping[ast.alias, IImport]
IArguments: TypeAlias = Mapping[ast.arg, IArgument]
IModuleCollection: TypeAlias = Mapping[str, IModuleSpec]
IDefOfImports: TypeAlias = IUseDefChains
IDunderAll: TypeAlias = Mapping[ast.Module, Collection[str] | None]
IMROs: TypeAlias = Mapping[ast.ClassDef, Sequence[ast.ClassDef | str]]

class AnalysisProvider(Generic[Tnode, Tresult], Protocol):
    def __call__(self, node:Tnode) -> Tresult:...

IAncestorsProvider: TypeAlias = AnalysisProvider[ast.Module, IAncestors]
IBenigetAnalysisProvider: TypeAlias = AnalysisProvider[IModuleSpec, IBenigetAnalysis]
IUseDefChainsProvider: TypeAlias = AnalysisProvider[IDefUseChains, IUseDefChains]
IUnreachableProvider: TypeAlias = AnalysisProvider[Tuple[ast.Module, IMinimalProjectDefs], IUnreachable]
IInstanceVarsProvider: TypeAlias = AnalysisProvider[Tuple[ast.Module, IDefUseChains], IInstanceVars]
IResolvedImportsProvider: TypeAlias = AnalysisProvider[IModuleSpec, IResolvedImports]
IArgumentsProvider: TypeAlias = AnalysisProvider[ast.Module, IArguments]

IDefOfImportsProvider: TypeAlias = AnalysisProvider[IMinimalProjectDefs, IDefOfImports]
IDunderAllProvider: TypeAlias = AnalysisProvider[IModuleCollection, IDunderAll]
IMROsProvider: TypeAlias = AnalysisProvider[IModuleCollection, IMROs]

## 

_not_registered: Any = object()

@dataclass(kw_only=True)
class AnalysisRegistry:
    """
    A registry for analyses.
    """
    
    # declare all analyses
    beniget:            IBenigetAnalysisProvider = field(default=_not_registered)
    ancestors:          IAncestorsProvider      = field(default=_not_registered)
    usechains:          IUseDefChainsProvider   = field(default=_not_registered)
    imports:            IResolvedImportsProvider = field(default=_not_registered)
    unreachable:        IUnreachableProvider    = field(default=_not_registered)
    instance_vars:      IInstanceVarsProvider   = field(default=_not_registered)
    arguments:          IArgumentsProvider      = field(default=_not_registered)

    def_of_imports:     IDefOfImportsProvider   = field(default=_not_registered)
    dunder_all:         IDunderAllProvider      = field(default=_not_registered)
    mros:               IMROsProvider           = field(default=_not_registered)

    def validate(self):
        analyses = tuple(asdict(self).items())
        if any(f is _not_registered for _,f in analyses):
            missing = ', '.join(n for n,f in analyses if f is _not_registered)
            raise RuntimeError(f'Incomplete registry, missing: {missing}')
    
    def register(self, analysis: str, provider: Type[AnalysisProvider]) -> None:
        setattr(self, analysis, provider(self))