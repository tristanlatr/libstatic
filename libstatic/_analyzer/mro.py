from __future__ import annotations

import ast
from typing import List, Mapping, Sequence, cast

from .._lib.model import Cls, Def
from .._lib.shared import unparse
from .._lib.process import TopologicalProcessor
from .._lib.exceptions import StaticAmbiguity, StaticException, StaticTypeError, NodeLocation
from .._lib.c3linear import c3_merge

from .state import MutableState, State

class ComputeClassMRO:
    def __init__(self, state:State, 
                 builder:TopologicalProcessor[Cls, Cls, Sequence[Cls|str]]) -> None:
        self.state = state
        self.builder = builder

    def get_bases(self, node: Cls|str, allow_ambiguity:bool=False) -> Sequence[Cls|str]:
        if isinstance(node, str):
            return []
        bases: List[Cls|str] = []
        # resolve bases, leave a warning if a resolved base is ambiguous
        for basenode in node.node.bases:
            if isinstance(basenode, ast.Subscript):
                basenode = basenode.value
            try:
                basedef: Cls|str = self.state.goto_definition(basenode, # type:ignore
                                    raise_on_ambiguity=not allow_ambiguity,
                                    follow_aliases=True)
                if not isinstance(basedef, Cls):
                    raise StaticTypeError(basedef, expected='Class', 
                                          filename=self.state.get_filename(cast(Def, basedef)))
            except StaticAmbiguity as e:
                self.state.msg(f'base is class is ambiguous: {str(e)}', ctx=basenode, thresh=1)
                return self.get_bases(node, allow_ambiguity=True)
            except StaticException as e:
                self.state.msg(f'cannot resolve base: {str(e)}', ctx=basenode, thresh=1)
                basedef = self.state.expand_expr(basenode) or unparse(basenode)
            else:
                # ensure that bases are processed before subclasses.
                self.builder.getProcessedObj(basedef)
                if self.builder.processing_state[basedef] is TopologicalProcessor.PROCESSING:
                    self.state.msg(f'cycle detected in base class {unparse(basenode)!r}'
                                   f' <-> {NodeLocation.make(basedef, filename=self.state.get_filename(basedef))}', 
                                   ctx=basenode)
                    # break the cycle with a string value
                    basedef = self.state.get_qualname(basedef)
            bases.append(basedef)
        return bases

    def compute_mro(self, node:Cls) -> Sequence[Cls|str]:
        """
        Return a list of classes in order corresponding to Python's MRO.
        """
        def get_computed_mro(o:Cls|str) -> Sequence[Cls|str]:
            if isinstance(o, str):
                return []
            else:
                return self.builder.result[o]

        result = [node]
        bases = self.get_bases(node)
        if not bases:
            return result
        else:
            return result + c3_merge(*(get_computed_mro(kls) for kls in bases), bases)

def compute_mros(state: MutableState) -> Mapping[Cls, Sequence[Cls|str]]:
    """
    Returns a mapping from class to it's resolved MRO.
    """

    class MROProcessor(TopologicalProcessor[Cls, Cls, Sequence[Cls|str]]):
        def getObj(self, key: Cls) -> Cls:
            return key

        def processObj(self, obj: Cls) -> Sequence[Cls|str]:
            return ComputeClassMRO(state, self).compute_mro(obj)

    return MROProcessor().process(c for c in 
            state._def_use_chains.values() if isinstance(c, Cls))
