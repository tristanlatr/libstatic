import ast
import sys
from pathlib import Path

from .model import Project
from .loader import load_path

paths = sys.argv[1:]
proj = Project(verbosity=1)
assert paths, 'missing positional argument: PATH, [PATH]'

for path in paths:
    p = Path(path)
    assert p.exists(), f'source path does not exist: {p}'
    load_path(proj, p)

proj.analyze_project()
total_defs = len(proj.state._def_use_chains)
mod_defs = len([o for o in proj.state._def_use_chains if isinstance(o, ast.Module)])

print(f'Project has {total_defs} use or definitions in the chains')
print(f'Project has {mod_defs} modules in the chains')
