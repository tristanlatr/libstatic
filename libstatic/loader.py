"""
Load a project form a python package/module path in the filesystem.
"""
from functools import partial
from pathlib import Path
import ast
import sys
from typing import Set, Tuple


from libstatic.model import Project

def _parse_file(path: Path) -> ast.Module:
    """Parse the contents of a Python source file."""
    with open(path, 'rb') as f:
        src = f.read() + b'\n'
    return _parse(src, filename=str(path))

if sys.version_info >= (3,8):
    _parse = partial(ast.parse, type_comments=True)
else:
    _parse = ast.parse

def _load_path(project:Project, 
                  path:Path, 
                  added:Set[Path], 
                  parent:Tuple[str,...]=()) -> None:
    if path in added:
        return
    if path.is_dir():
        init_file = (path / '__init__.py')
        if not init_file.is_file():
            return
        mod = _parse_file(init_file)
        curr_pack = parent + (path.name,)
        project.add_module(mod, '.'.join(curr_pack), 
                           is_package=True,
                           filename=init_file.as_posix())
        added.add(init_file)
        for p in sorted(path.iterdir()):
            _load_path(project, p, added, parent=curr_pack)
    elif path.is_file() and path.suffix == '.py':
        mod = _parse_file(path)
        project.add_module(mod, '.'.join(parent + (path.stem,)), 
                           filename=path.as_posix())
        added.add(path)

def load_path(project:Project, path:Path) -> None:
    """
    Load a project form a python package/module path in the filesystem.
    Project.analyze_project() must still be called after loading a path into the project.

    >>> p = Project()
    >>> load_path(p, Path('./libstatic'))
    >>> p.analyze_project()
    ... ...
    """
    _load_path(project, path, set(), ())