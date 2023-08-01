import ast
import sys
from pathlib import Path
import argparse
import time

from .model import Project
from .loader import load_path
from .exceptions import StaticNameError

def location(node:ast.AST, filename:str) -> str:
    return StaticNameError(node, filename=filename).location()

def main():
    parser = argparse.ArgumentParser(description='',
                                     prog='python3 -m libstatic',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('PATH', help='path to python modules/packages', nargs='+')
    parser.add_argument('-v', '--verbose', dest='verbosity', action='count', help='increase verbosity', )
    parser.add_argument('-d', '--dependencies', action='store_true', help='load available dependencies')
    parser.add_argument('-u', '--uses', default=None, help='find usages of the definitions with the given qualname', nargs='+')
    
    args = parser.parse_args()

    proj = Project(verbosity=args.verbosity or 0, 
                   nested_dependencies=16 if args.dependencies else 0, 
                   python_version=tuple(sys.version_info[:2]))

    for path in args.PATH:
        p = Path(path)
        assert p.exists(), f'source path does not exist: {p}'
        load_path(proj, p)

    proj.analyze_project()
    total_defs = len(proj.state._def_use_chains)
    mod_defs = len([o for o in proj.state._def_use_chains if isinstance(o, ast.Module)])

    print(f'Project has {total_defs} use or definitions in the chains')
    print(f'Project has {mod_defs} modules in the chains')

    t0 = time.time()
    if args.uses:
        for qualname in args.uses:
            for i,d in enumerate(proj.state.get_defs_from_qualname(qualname)):
                usages = list(proj.state.goto_references(d))
                qualname_id = f'{qualname!r}{"" if i==0 else f" ({i})"} at {location(d.node, proj.state.get_filename(d.node))}'
                if usages:
                    print(f'Found {len(usages)} usages of {qualname_id} in the project')
                    for use in usages:
                        print(f' - {location(use.node, proj.state.get_filename(use.node))}')
                else:
                    print(f'Did not found any use of {qualname_id}')
    
    t1 = time.time()
    proj.state.msg(f"loading references took {t1-t0} seconds", thresh=1)
if __name__ == "__main__":
    main()