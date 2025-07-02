# libstatic = Framework for syntax tree analysis and manipulation
# psta = Python syntax tree transformer and analyzer

"""
Syntax tree transformer and analyzer

-- run, troubleshoot and showcase analyses and transformations --  

SOURCE_PATHS / --source - The source files or directories. 

--run=<pass function>[(<option>=<value>,<option>=<value>, ...)] (repeatable)
--result=<pass function>[(<option>=<value>,<option>=<value>, ...)]:[filename.]<extension> (repeatable)

It will run the given passes on all discovered trees under the SOURCE_PATHS 
and write analyses result files under the same relative locations. 

If the resulting pass is a forest-wide analysis, 
the filename must also be given alongside with the extension.

--setup=<instrumentation function>[(<option>=<value>,<option>=<value>, ...)] (repeatable)

A instrumentation function is a function that expose a signature like: 
``(pm: PassManager)`` or ``(pm: PassManager, *, options=False)`` . 
This function will be called once, after the passmanger is created and before
any trees is added. It is usually used to install hooks, but can also patch/overload
some parts of the given PassManager instance, or even pre-load some trees.

See PassManager.hooks for more information about how to use hooks. 

-c/--config=<file> - Read config from a file. Argument overrides config values.
    Supports both TOML and INI formats, it looks for pipeline definitions
    under the following names: 
        'tool:libstatic[:<name>]', 'tool.libstatic[.<name>]'. 
    By default it reads the first file found: libstatic.ini, pyproject.toml or setup.cfg. 
    
-e/--section=<name>[,<name2>] - The name(s) of the sections(s) in the config file to load, 
    a single config file can contains several sections with in each, a pipeline definition. 
    This argument can't be set from config file since it'a a meta option.

    Configuration file is IGNORED if option -e=<name> is not passed.
    Attention!! Repeatable options are overriden by arguments, so if
    your config file defined a pipeline run option, any --run argument will completely 
    override the list from the config file.
    
-o/--output=<directory> - Output dir.

-v/--verbosity=[<logger name>:]<level> - DEBUG/INFO/WARNING/ERROR/CRITICAL (repeatable)
-l/--logfile=[<logger name>:]<filename> - A filename to log into (repeatable), the default behaviour
    will log any WARNING or worse to standard error.
    If 'logger name' is not provided, it defaults to 'libstatic'. 
    Under the hood, this is translated to a dict loaded with logging.config.dictConfig().
-W/--warning-as-error - If the program emits a WARNING message, the exit code will be 3.
    In any case, if the program emits an ERROR or CRITICAL message, the exit code will be 2.

# --get-children=<function name> = The function to use to iter the children of a node: 
#     (node: Node) -> Iterable[Node]

--load-trees=<function name> - The function to use to load Tree instances from
    the given source file or directory. 
    So it's signture is: ``(given: Path) -> Iterable[Tree]``, 
    the default is `libstatic.finder.load_ast`.

    Example of config file using doctils tree:: 
    ; Pipeline 'makehtml'
    [tool:libstatic:makehtml]
    source=
        ./docs/
    # get-children=libstatic.contrib.docutils.get_children
    load-tree=libstatic.contrib.docutils.load_docutils
    setup=
        libstatic.instrumentations.statistics
        libstatic.instrumentations.check_analyses_side_effects
        libstatic.contrib.docutils.enforce_docutils_attributes
    run=
        libstatic.contrib.docutils.remove_unsupported
    result=
        libstatic.contrib.docutils.generate_html    ::  index.html
    output=./build/www
        
Example of config file::
    ; Pipeline 'transform'
    [tool:libstatic:transform]
    source=
        ./src/my_project
    setup=
        libstatic.instrumentations.statistics
        libstatic.instrumentations.check_analyses_side_effects
    run=
        libstatic.transformations.remove_dead_code
        libstatic.transformations.remove_legacy_code(python_version=(3,11))
        libstatic.transformations.add_dependent_modules(depth=3, exclude=['my_project.vendored.*'])
        libstatic.transformations.normalize__all__
        libstatic.transformations.expand_wildcards
        libstatic.transformations.fold_constants
        libstatic.transformations.undataclass
    result=
        libstatic.instrumentations.dump_statistics  ::    stats.txt
        libstaic.analyses.unparse:  {tree}.py
    
    output=./build/code
    verbosity=
        libstatic.transformations:DEBUG
    logfile = ./logs/logs.txt

    ; Pipeline 'graphs'
    [tool:libstatic:graphs]
    source=
        ./src/my_project
    result=
        my_passes.cfg(entry_point='my_lib.client.Client.get_thing', format='dot')   ::  get_thing_cfg.dot
        my_passes.call_graph(entry_point='my_lib.client.Client.get_thing', format='dot')    ::   get_thing_call_graph.dot
        libstatic.analyses.import_graph(format='dot')   ::  import_graph.dot
    verbosity=
        DEBUG
    logfile = 
        libstatic.passmanager:./logs/passmanager_logs.txt
        libstatic:./logs/all_logs.txt
    output=./build/graphs

    ; we could even abuse it an do something like:
    run = 
        libstatic.instrumentations.repeat_until_fixed_point('my_passes.optimize_pass', up_to=6)

Example of CLI usage::
    psta -e transform,graphs \ # select some pipelines
        -v DEBUG -l logs/log.txt # override some options

Looks like tox? Well... that's on purpose.
    
Without a configuration file::
    pystam --setup=libstatic.intrumentations.statistics \
        --run 'libstatic.transformations.remove_legacy_code(python_version=(3,11))' \
        --run 'libstatic.transformations.add_dependent_modules(depth=3, exclude=["my_project.vendored.*"])' \
        --run libstatic.transformations.undataclass [...] \
        --result 'libstaic.analyses.unparse::{tree}.py' \
        --output build/code \
        -v DEBUG -l logs/log.txt \
        ./src/my_project

"""

import configparser, argparse

# the argument parsing process is a little bit tricky,
# but this is necessay to support tox-like sections. 
# first we parse know arguments with only option -e
# then we dynamically build the argument parser instance based on the 
# provided section and parse all argument/config file accordingly.

# a "pipeline" is defined by these three elements:
# - the list of instrumentation setup functions
# - the list of passes to run
#   This is not obvioius since a forest-wide pass can be inserted
#   at any point in the list, so basically a list of passes like:
#   - tree-wide pass1
#   - forest-wide pass1
#   - tree-wide pass2
#   - tree-wide pass3
#   - forest-wide pass2
#   - tree-wide pass4
#   should be converted to a Plan object that represents the following process:
#   - run tree-wide pass1 on every tree
#   - run forest-wide pass1
#   - run tree-wide pass2 and pass3 on every tree
#   - run forest-wide pass2
#   - run tree-wide pass4 on every tree
#   Ideally, the Plan should be a Pass like others, runnable just like others.
# - the list of resulting passes, that also ends-up in a Plan
#   but has comlementary informations regarding output extension and filenames.
# for each element parse list of strings into the right object kinds
# this is non-trivial and we might just use something eval-like. 

# the logger configurations are defined by:
# - a mapping from logger name to the verbosity level, WARNING is used by default.
# - a mapping from logger name to a logfile, sys.stderr is used by default.
# - whether warning-as-error is enabled.

# what's left is the source paths and the output directory. 
