import ast
import sys
from pathlib import Path
import argparse
import time
from typing import Optional

from .model import Project
from .loader import load_path
from .exceptions import StaticNameError

def location(node:ast.AST, filename:Optional[str]) -> str:
    return StaticNameError(node, filename=filename).location()

def main() -> None:
    parser = argparse.ArgumentParser(description='Playground frontend for the library.',
                                     prog='python3 -m libstatic',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('PATH_or_MODULE_NAME', help='path or fully qualified name of python modules/packages, '
                        'in the case of a module name only modules avalaible with typeshed_client are collected.', nargs='+')
    parser.add_argument('-v', '--verbose', dest='verbosity', action='count', help='increase verbosity', )
    parser.add_argument('-d', '--dependencies', action='store_true', help='load available dependencies')
    parser.add_argument('-u', '--uses', default=None, help='find usages of the definitions with the given qualname', nargs='+')
    parser.add_argument('--dir', default=None, help='list locals of specified definitions', nargs='+')
    
    args = parser.parse_args()

    proj = Project(verbosity=args.verbosity or 0, 
                   nested_dependencies=16 if args.dependencies else 0, 
                   python_version=tuple(sys.version_info[:2]))

    for path_or_modname in args.PATH_or_MODULE_NAME:
        p = Path(path_or_modname)
        if not p.exists():
            # check if it's a typeshed module
            assert all(n.isidentifier() for n in path_or_modname.split('.')), \
                f'file {path_or_modname} doesn\'t exist'
            assert proj.add_typeshed_module(path_or_modname), \
                f'stubs for module {path_or_modname} not found and file ./{path_or_modname} doesn\'t exist'
        else:
            load_path(proj, p)

    proj.analyze_project()
    total_defs = len(proj.state._def_use_chains)
    mod_defs = len([o for o in proj.state._def_use_chains if isinstance(o, ast.Module)])

    print(f'Project has {total_defs} use or definitions in the chains')
    print(f'Project has {mod_defs} modules in the chains')

    if proj.options.verbosity>0:
        for mod in proj.state.get_all_modules():
            print(f' - {mod.name()}')

    if args.uses:
        t0 = time.time()
        for qualname in args.uses:
            for i,d in enumerate(proj.state.get_defs_from_qualname(qualname)):
                usages = list(proj.state.goto_references(d)) # type:ignore
                qualname_id = f'{qualname!r}{"" if i==0 else f" ({i})"} at {location(d.node, proj.state.get_filename(d.node))}'
                if usages:
                    print(f'Found {len(usages)} usages of {qualname_id} in the project')
                    for use in usages:
                        print(f' - {location(use.node, proj.state.get_filename(use.node))}')
                else:
                    print(f'Did not found any use of {qualname_id}')
        t1 = time.time()
        proj.state.msg(f"loading references took {t1-t0} seconds", thresh=1)
    
    if args.dir:
        for qualname in args.dir:
            for i,d in enumerate(proj.state.get_defs_from_qualname(qualname)):
                qualname_id = f'{qualname!r}{"" if i==0 else f" ({i})"} at {location(d.node, proj.state.get_filename(d.node))}'
                print(f'Locals of {qualname_id}')
                for locs in proj.state.get_locals(d).values():
                    locs = list(filter(None, locs))
                    loc = locs[0]
                    locs_repr = ', '.join(repr(l) for l in locs)
                    if proj.options.verbosity>0:
                        print(f' - {locs_repr} at {location(loc.node, proj.state.get_filename(loc.node))}') # type: ignore
                    else:
                        print(f' - {locs_repr}')
    
if __name__ == "__main__":
    main()