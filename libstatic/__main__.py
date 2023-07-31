import ast
import sys
from pathlib import Path
import argparse

from .model import Project
from .loader import load_path

def main():
    parser = argparse.ArgumentParser(description='',
                                     prog='python3 -m libstatic',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('PATH', help='path to python modules/packages', nargs='+')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='increase verbosity', )
    parser.add_argument('-d', '--dependencies', default=0, type=int, help='try to load nested dependencies')
    args = parser.parse_args()

    proj = Project(verbosity=args.verbosity, 
                   nested_dependencies=args.dependencies, 
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

if __name__ == "__main__":
    main()