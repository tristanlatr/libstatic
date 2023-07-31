import ast
from typing import List, Set, Dict, Any, Mapping, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

from .model import State, Options
from .shared import LocalStmtVisitor

def get_unreachable(state: State, options: Options, mod: ast.Module) -> Set[ast.AST]:
    known_values: Dict[str, Any] = {}
    version = options.python_version
    if version:
        assert isinstance(version, tuple)
        assert len(version) >= 2
        assert all(isinstance(p, int) for p in version)

        known_values["sys.version_info"] = version
        known_values["sys.version_info.major"] = version[0]
        known_values["sys.version_info.minor"] = version[1]

    return _Unreachable(state, known_values).visit_Module(mod)

class HasBody(Protocol):
    body: List[ast.stmt]
class HasOrelse(HasBody, Protocol):
    orelse: List[ast.stmt]

class _Unreachable(LocalStmtVisitor):
    def __init__(self, state: State, known_values: Mapping[str, Any]) -> None:
        self._state = state
        self._known_values: Mapping[str, Any] = known_values
        self._unreachable_nodes: Set[ast.AST] = set()

    def visit_If(self, node: ast.If) -> None:
        try:
            testval = self._state.literal_eval(
                node.test, 
                known_values=self._known_values, 
                raise_on_ambiguity=True
            )
        except Exception:
            self.generic_visit(node)
        else:
            unreachable = node.orelse if testval else node.body
            reachable = node.body if testval else node.orelse
            mark_unreachable = _MarkUnreachable(self._unreachable_nodes)
            for n in unreachable:
                self._state.msg(f"is unreachable.", ctx=n, thresh=1)
                mark_unreachable.visit(n)
            for n in reachable:
                self.generic_visit(n)

    def visit_Try(self, node:ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)
        # TODO: mark node in handlers as "exceptionally reachable"
        for h in node.handlers:
            self.generic_visit(h)

    def visit_Module(self, node: ast.Module) -> Set[ast.AST]:
        self.generic_visit(node)
        return self._unreachable_nodes


class _MarkUnreachable(ast.NodeVisitor):
    def __init__(self, unreachable: "set[ast.AST]") -> None:
        self._unreachable = unreachable

    def visit(self, node: ast.AST) -> None:
        self._unreachable.add(node)
        self.generic_visit(node)
