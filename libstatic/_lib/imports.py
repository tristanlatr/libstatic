import ast
from typing import Any, Dict, Mapping, Optional, Tuple, NamedTuple, Union


class ImportInfo(NamedTuple):
    orgmodule: str
    orgname: Optional[str] = None


class ImportParser(ast.NodeVisitor):
    """
    Transform import statements into a mapping from `ast.alias` to `ImportInfo`.
    One instance of `ImportParser` can be used to parse all imports in a given module.
    """

    def __init__(self, modname: str, *, is_package: bool) -> None:
        self._modname = tuple(modname.split("."))
        self._is_package = is_package
        self._result: Dict[ast.alias, ImportInfo] = {}

    # parsing imports, partially adjusted from typeshed_client

    def generic_visit(self, node: ast.AST) -> Any:
        raise TypeError()

    def visit(self, node: ast.AST) -> Mapping[ast.alias, ImportInfo]:
        return super().visit(node)  # type: ignore

    def visit_Import(self, node: ast.Import) -> Mapping[ast.alias, ImportInfo]:
        self._result.clear()
        for al in node.names:
            if al.asname:
                self._result[al] = ImportInfo(orgmodule=al.name)
            else:
                # here, we're lossng the dependency on "driver" of "import pydoctor.driver" in 'orgmodule',
                self._result[al] = ImportInfo(orgmodule=al.name.split(".", 1)[0])
        return self._result

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Mapping[ast.alias, ImportInfo]:
        self._result.clear()
        current_module: Tuple[str, ...] = self._modname
        module: Tuple[str, ...]

        if node.module is None:
            module = ()
        else:
            module = tuple(node.module.split("."))
        if node.level == 0:
            source_module = module
        else:
            if node.level == 1:
                if self._is_package:
                    relative_module = current_module
                else:
                    relative_module = current_module[:-1]
            else:
                if self._is_package:
                    relative_module = current_module[: 1 - node.level]
                else:
                    relative_module = current_module[: -node.level]

            if not relative_module:
                raise ValueError(
                    "relative import level (%d) too high" % node.level,
                )

            source_module = relative_module + module

        for alias in node.names:
            self._result[alias] = ImportInfo(
                orgmodule=".".join(source_module), orgname=alias.name
            )

        return self._result


class ParseImportedNames(ast.NodeVisitor):
    """
    Maps each `ast.alias` to their `ImportInfo` counterpart.
    """

    def __init__(self, modname: str, *, is_package: bool) -> None:
        super().__init__()
        self._import_parser = ImportParser(modname, is_package=is_package)

    def visit_Module(self, node: ast.Module) -> Mapping[ast.alias, ImportInfo]:
        self._result: Dict[ast.alias, ImportInfo] = {}
        self.generic_visit(node)
        return self._result

    def generic_visit(self, node: Any) -> None:
        if isinstance(node, ast.expr):
            return
        else:
            super().generic_visit(node)

    def visit_Import(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        try:
            imports = self._import_parser.visit(node)
        except ValueError:
            return
        for al, info in imports.items():
            self._result[al] = info

    visit_ImportFrom = visit_Import
