"""
Python's scoping rules, as code.

It does not cover all invalid cases and might get confused by python runtime semantics. 
More informations on gvanrossum's blog: U{https://github.com/gvanrossum/gvanrossum.github.io/blob/main/formal/scopesblog.md} 
and U{https://github.com/gvanrossum/gvanrossum.github.io/blob/main/formal/scopes.md}

There are lots of invariants and ideas not yet expressed in code:

- scopes form a tree with a GlobalScope at the root
- there are no GlobalScopes elsewhere in the tree
- *using* a name before a nonlocal declaration is also an error
- a way to check a scope's invariants
- locals/nonlocals/globals are disjunct
- everything about comprehensions
- translating the AST into a tree of scopes
- Using a subset of Python

"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import types
from typing import Iterable, Iterator

class Scope:
    scope_name: str
    parent: Scope | None
    uses: set[str]
    locals: set[str]
    nonlocals: set[str]
    globals: set[str]

    def __init__(self, scope_name: str, parent: Scope | None):
        self.scope_name = scope_name
        self.parent = parent
        self.uses = set()
        self.locals = set()
        self.nonlocals = set()
        self.globals = set()
        # locals, nonlocals and globals are all disjunct

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.scope_name!r})"

    def store(self, name: str) -> None:
        if name in self.locals or name in self.nonlocals or name in self.globals:
            return
        self.locals.add(name)

    def load(self, name: str) -> None:
        self.uses.add(name)

    def add_nonlocal(self, name: str) -> None:
        if name in self.uses:
            raise SyntaxError("name used prior to nonlocal declaration")
        if name in self.locals:
            raise SyntaxError("name assigned before nonlocal declaration")
        if name in self.globals:
            raise SyntaxError("name is global and nonlocal")
        self.nonlocals.add(name)

    def add_global(self, name: str) -> None:
        if name in self.uses:
            raise SyntaxError("name used prior to global declaration")
        if name in self.locals:
            raise SyntaxError("name assigned before global declaration")
        if name in self.nonlocals:
            raise SyntaxError("name is nonlocal and global")
        self.globals.add(name)

    def global_scope(self) -> GlobalScope:
        # GlobalScope overrides this
        assert self.parent is not None
        return self.parent.global_scope()

    def enclosing_closed_scope(self) -> ClosedScope | None:
        if self.parent is None:
            return None
        elif isinstance(self.parent, ClosedScope):
            return self.parent
        else:
            return self.parent.enclosing_closed_scope()

    def lookup(self, name: str) -> Scope | None:
        # Implemented differently in OpenScope, GlobalScope and ClosedScope
        raise NotImplementedError


class OpenScope(Scope):
    def lookup(self, name: str) -> Scope | None:
        if name in self.locals:
            return self
        else:
            s = self.enclosing_closed_scope()
            if s is not None:
                return s.lookup(name)
            else:
                return self.global_scope().lookup(name)


# module scope, there used to be a TopLevelScope, but it messed up with the lookup :/
class GlobalScope(OpenScope):
    parent: None  # Must be None

    def __init__(self):
        super().__init__("<globals>", None)

    def global_scope(self) -> GlobalScope:
        return self

    def lookup(self, name: str) -> Scope | None:
        if name in self.locals:
            return self
        else:
            return None

    def add_nonlocal(self, name: str) -> None:
        raise SyntaxError("nonlocal declaration not allowed at module level")

    def add_global(self, name: str) -> None:
        return self.store(name)



class ClassScope(OpenScope):
    parent: Scope  # Cannot be None

    def __init__(self, name: str, parent: Scope):
        super().__init__(name, parent)
        parent.store(name)


class ClosedScope(Scope):
    parent: Scope  # Cannot be None

    def lookup(self, name: str) -> Scope | None:
        if name in self.locals:
            return self
        elif name in self.globals:
            return self.global_scope()
        else:
            res: Scope | None = None
            p: Scope | None = self.enclosing_closed_scope()
            if p is None:
                res = None
            else:
                res = p.lookup(name)
            if name in self.nonlocals and not isinstance(res, ClosedScope):
                # res could be None or GlobalScope
                raise SyntaxError(f"nonlocal name {name!r} not found")
            elif res is not None:
                return res
            else:
                # changed here: closed scopes have access to the globals
                return self.global_scope().lookup(name) 


class FunctionScope(ClosedScope):
    def __init__(self, name: str, parent: Scope):
        super().__init__(name, parent)
        parent.store(name)


class LambdaScope(FunctionScope):
    pass


class ComprehensionScope(ClosedScope):
    pass


# Builder

LOAD = ast.Load()
STORE = ast.Store()


class Builder:
    """Build scope structure from AST."""

    globals: GlobalScope
    scopes: dict[ast.AST, Scope]
    _current: Scope

    def store(self, name: str) -> None:
        self._current.store(name)

    @contextlib.contextmanager
    def push(self, node: ast.AST, scope: Scope) -> Iterator[Scope]:
        parent = self._current
        try:
            self._current = scope
            self.scopes[node] = scope
            yield scope
        finally:
            self._current = parent

    def build(self, node: object | None) -> None:
        match node:
            case (
                None
                | str()
                | bytes()
                | bool()
                | int()
                | float()
                | complex()
                | types.EllipsisType()
            ):
                pass
            case list():
                for n in node:
                    self.build(n)
            case ast.Name(id=name, ctx=ast.Store()):
                self.store(name)
            case ast.Name(id=name, ctx=ast.Load()):
                self._current.load(name)
            case ast.Nonlocal(names=names):
                for name in names:
                    self._current.add_nonlocal(name)
            case ast.Global(names=names):
                for name in names:
                    self._current.add_global(name)
            case ast.ImportFrom(names=names):
                for a in names:
                    if a.asname:
                        self.store(a.asname)
                    elif a.name != "*":
                        self.store(a.name)
            case ast.Import(names=names):
                for a in names:
                    if a.asname:
                        self.store(a.asname)
                    else:
                        name = a.name.split(".")[0]
                        self.store(name)
            case ast.ExceptHandler(type=typ, name=name, body=body):
                self.build(typ)
                if name:
                    self.store(name)
                self.build(body)
            case ast.MatchAs(name=name) | ast.MatchStar(name=name):
                if name:
                    self.store(name)
            case ast.Lambda(args=args, body=body) as node:
                self.build(args)  # defaults
                with self.push(node, LambdaScope("<lambda>", self._current)):
                    self.build(body)
            case ast.NamedExpr(target=target, value=value):
                # TODO: Various other forbidden cases from PEP 572,
                # e.g. [i := 0 for i in a] and [i for i in (x := a)].
                assert isinstance(target, ast.Name)
                self.build(value)
                s = self._current
                while isinstance(s, ComprehensionScope):
                    s = s.parent
                if isinstance(s, ClassScope):
                    raise SyntaxError("walrus in comprehension cannot target class")
                s.store(target.id)
            case ast.comprehension(target=target, ifs=ifs):
                self.build(target)
                # node.iter is built by the next two cases
                self.build(ifs)
            case (ast.ListComp(elt=elt, generators=gens) | ast.SetComp(
                elt=elt, generators=gens
            ) | ast.GeneratorExp(elt=elt, generators=gens)) as node:
                self.build(gens[0].iter)
                name = f"<{node.__class__.__name__}>"
                with self.push(node, ComprehensionScope(name, self._current)):
                    self.build(elt)
                    self.build(gens)
                    self.build([g.iter for g in gens[1:]])
            case ast.DictComp(key=key, value=value, generators=gens) as node:
                self.build(gens[0].iter)
                with self.push(node, ComprehensionScope(f"<DictComp>", self._current)):
                    self.build(key)
                    self.build(value)
                    self.build(gens)
                    self.build([g.iter for g in gens[1:]])

            case ast.FunctionDef(
                name=name,
                args=args,
                body=body,
                decorator_list=decorator_list,
                returns=returns,
            ) as node:
                self.build(decorator_list)
                self.build(args)  # Annotations and defaults
                self.build(returns)
                with self.push(node, FunctionScope(name, self._current)):
                    for al in args.posonlyargs + args.args + args.kwonlyargs:
                        self.store(al.arg)
                    self.build(body)
                self.store(name)
            case ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=body,
                decorator_list=decorator_list,
            ) as node:
                self.build(decorator_list)
                self.build(bases)
                self.build(keywords)
                with self.push(node, ClassScope(name, self._current)):
                    self.build(body)
                self.store(name)
            case ast.Module(
                body=body
            ) as node:
                # init class attributes
                self.globals = GlobalScope()
                self._current = self.globals
                self.scopes = {node: self._current}
                self.build(body)
            case ast.AST():
                for _k, value in node.__dict__.items():
                    if not _k.startswith("_"):
                        self.build(value)
            case _:
                assert False, repr(node)


def depth(s: Scope) -> int:
    n = 0
    while s.parent is not None:
        n += 1
        s = s.parent
    return n


def expand_globs(filenames: list[str]) -> Iterator[str]:
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            import glob
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


tab = "  "

def dump(scopes: Iterable[Scope]) -> str:
    b = io.StringIO()
    for scope in scopes:
        indent = tab * depth(scope)
        print(f"{indent}{scope}: L={sorted(scope.locals)}", end="", file=b)
        if scope.nonlocals:
            print(f"; NL={sorted(scope.nonlocals)}", end="", file=b)
        if scope.globals:
            print(f"; G={sorted(scope.globals)}", end="", file=b)
        uses = {}
        for name in sorted(scope.uses):
            uses[name] = scope.lookup(name)
        print(f"; U={uses}", file=b)
    return b.getvalue()

def main():
    dump = False
    files = sys.argv[1:]
    if files and files[0] == "-d":
        dump = True
        del files[0]
    if not files:
        files.append(os.path.join(os.path.dirname(__file__), "default.py"))
    for file in expand_globs(files):
        print()
        print(file + ":")
        with open(file, "rb") as f:
            data = f.read()
        root = ast.parse(data)
        if dump:
            print(ast.dump(root, indent=2))
        b = Builder()
        b.build(root)
        print(dump(b.scopes.values()))

if __name__ == "__main__":
    main()