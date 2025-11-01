"""
Declares the passe objects and related tools.
"""
from __future__ import annotations

from functools import partial
from inspect import signature, Parameter
from itertools import chain

from typing import (
    Callable,
    Collection,
    Container,
    Final,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    overload,
)

import attrs

from libstatic._lib.structures import (
    FrozenDict,
    OrderedSet,
)

from ._model import Tree, Forest, Level, PassKind, Element

if TYPE_CHECKING:
    from _passmanager import Connector, CastableToDict

type PassLike = PassPrototype | PassInstance

class IPassFunction(Protocol):
    """
    A pass function is a two positional argument
    function with optionnaly any keyword arguments (aka "run options") 
    that returns something that can be converted to a dict.
    """

    def __call__(
        self, c: Connector, element: Element, **kwargs: Hashable
    ) -> CastableToDict: ...

@attrs.frozen()
class PassPrototype:
    """
    Carries all the meta information about a pass.

    Calling instances of this object will produce a L{PassInstance}.
    """

    #: The pass function
    do_pass: IPassFunction

    #: A name for this pass, this will be used in the
    #: dependencies attribute name if that's a analysis.
    name: str

    #: the type of pass: transformation or analysis.
    kind: PassKind
    
    #: on what kind of element this pass runs on? this is conceptual.
    runs_on: Level

    #: on what type of object this pass runs on?
    #: a isinstance check will be done and L{TypeError}
    #: will be raised for any mismatch.
    runs_on_type: type | tuple[type, ...]

    #: required parameters names declaration
    params: tuple[str, ...] = attrs.field(
        default=(), converter=tuple
    )  # at least an empty tuple

    #: options names to their default values declaration
    optional_params: Mapping[str, Hashable] = attrs.field(
        default=FrozenDict(), converter=FrozenDict
    )  # at least an empty map

    #: a sequence of dependecies that will be (lazily) bound to 
    #: attributes inside the 'deps' property of the connector.
    dependencies: tuple[PassLike | str, ...] = attrs.field(
        default=(), converter=tuple
    )  # at least an empty tuple

    #: keyword arguments passed to the pass decorator, these keywords
    #: are used a options for the plugins. 
    pass_options: Mapping[str, Any] = attrs.field(
        default=FrozenDict(), converter=FrozenDict
    )  # at least an empty map

    # Desperate attempt to make it work with doctests :/ not working
    @property
    def __doc__(self) -> str | None:
        return self.do_pass.__doc__

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def __wrapped__(self) -> IPassFunction:
        return self.do_pass

    def __str__(self) -> str:
        # i.e. "Node analysis 'def_use_chains'"
        return f"{self.runs_on.name.title()} {self.kind.name.lower()} {self.name!r}"

    @property
    def proto(self) -> PassPrototype:
        """
        Convenience to be able to call .proto on any PassLike.
        """
        return self

    def _replace(self, **kwargs: Any) -> PassPrototype:
        return attrs.evolve(self, **kwargs)

    def get_dependencies(self, get_passe: Callable[[str], PassLike]) -> Collection[PassLike]:
        """
        Get the direct dependencies of this pass.
        """
        return {get_passe(d) if isinstance(d, str) else d for d in self.dependencies}

    def get_all_dependencies(self, get_passe: Callable[[str], PassLike]) -> Collection[PassLike]:
        """
        Transitively iterate on all dependencies of this pass.
        """
        seen: set[PassLike] = OrderedSet()

        def _yield_deps(c: PassLike) -> Iterator[PassLike]:
            yield from (d for d in c.proto.get_dependencies(get_passe) if d not in seen)
            yield from (
                d
                for d in chain.from_iterable(
                    _yield_deps(dep) for dep in c.proto.get_dependencies(get_passe)
                )
                if d not in seen
            )

        seen.update(_yield_deps(self))
        return seen

    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        return self._instanciate()(*args, **kwargs)

    def missing_param(self) -> str | None:
        """
        Whether this pass prototype requires any parameter. 
        """
        return self._instanciate().missing_param()

    def _instanciate(self) -> PassInstance:
        return PassInstance(self, FrozenDict())


_nah = object()


@attrs.frozen()
class PassInstance:
    """
    I represent a instance of a L{PassPrototype}.

    Whereas the pass prototype, instance of this class carries the actual
    values of the parameters in use (). If the prototype do not declare any parameters,
    then this representation is redundant.

    A pass instance have no knowledge of the node it will run onto. the pass
    instance describe the concrete behaviour of a pass, and can be run several
    times on several elements.
    """

    proto: PassPrototype
    args: FrozenDict[str, Hashable]

    def __call__(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        newpass = self
        if args or kwargs:
            newpass = newpass._add_args(*args, **kwargs)
        return newpass

    def _add_args(self, *args: Hashable, **kwargs: Hashable) -> PassInstance:
        params = self.proto.params
        optional_params = self.proto.optional_params
        len_optinals = len(optional_params)
        len_required = len(params)

        if len(kwargs) > (len_optinals + len_required):
            raise TypeError(
                "too many keyword parmeters, expected at most "
                f"{len_optinals + len_required} keywords"
            )
        if len(args) > len_required:
            raise TypeError(
                "too many positional parmeters, expected at most "
                f"{len_required} positionals and {len_required} keywords"
            )

        self_args = self.args
        self_args_get = self_args.get
        args_dict = {}

        # support passing required parameters as positionals
        for pname, value in zip(params, args):
            # This prevents the creation of new instance of
            # PassInstance whith the same params values.
            if value != (self_args_get(pname, _nah)):
                args_dict[pname] = value

        for pname, value in tuple(kwargs.items()):
            if (pname not in params) and (pname not in optional_params):
                raise TypeError(f"unexpected argument {pname!r}")
            if pname in args_dict:
                raise TypeError(f"got several values for parameter {pname!r}")
            if value != self_args_get(pname, _nah):
                args_dict[pname] = value

        if args_dict:
            return PassInstance(self.proto, 
                                FrozenDict({**self.args, **args_dict}))
        else:
            return self

    def missing_param(self) -> str | None:
        """
        Whether this pass instance is missing a required parameter.
        """
        if any((missing := p) not in self.args for p in self.proto.params):
            return missing
        return None

    def _replace(self, **kwargs: Any) -> PassInstance:
        return attrs.evolve(self, **kwargs)


_posargs = frozenset(
    (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.POSITIONAL_ONLY,
    )
)
_unsupportedargs = frozenset(
    (
        Parameter.VAR_KEYWORD,
        Parameter.VAR_POSITIONAL,
    )
)

_on_cls_2_level = {Forest: Level.FOREST, Tree: Level.TREE}


def new_pass_prototype(
    do_pass: Callable[..., CastableToDict],
    *,
    kind: PassKind,
    on: type | Iterable[type],
    dependencies: Sequence[PassLike] | None = None,
    **options: Any, 
    # memoize: bool = True,  # for analyses only
    # immutable: bool = False,  # for analyses only
    # file_cached: bool = False,
) -> PassPrototype:
    """
    Create a pass prototype from a given callable and a handful of options.

    @param do_pass: A callable that contains the driving logic of your pass.
      By convention, the callable should be either

       - a generator function yielding tuples: (key, value)
       - a function returning a dict

      A passe can provide metadata that are not directly
      meant to be presented to the users but rather use to internally optimize runs.

      See L{AnalysisReturn} or L{TransformationReturn} 
      for supported keys and value types of the returned mapping.

      Support two variants:

        - no parameter -> two posargs::
            def f(c, node): ...
        - with parameters -> two posargs and x keywords::
            def f(c, node, *, arg1, arg2=False): ...

    @param kind: L{ANALYSIS} or L{TRANSFORMATION}.
    @param on: The type of the element the pass is supposed to be run on.
        I.e. L{Forest}, L{Tree}, L{ast.Module}, L{ast.FunctionDef}.
        This can also be a tuple of types, but this is only applicable
        if your pass runs on syntax tree nodes (not L{Tree} or L{Forest}).
    @param dependencies: Sequence of the pass-like dependencies of this pass.
   
    """
    try:
        do_pass_sig = signature(do_pass)
    except Exception as e:
        raise TypeError(
            "This functions is not supported at the moment "
            f"because it can't be inspected: {do_pass}"
        ) from e

    # Validate the signature and extract parameters
    params: list[str] = []  # names
    optional_params: dict[str, Hashable] = {}  # names to their defaults

    nb_pos_args = 0
    for param_name, param in do_pass_sig.parameters.items():
        if param.kind in _posargs:  # we have a positional argument
            nb_pos_args += 1
            if nb_pos_args > 2:
                raise TypeError(
                    "A pass function must not take more "
                    "than two positional arguments, please use keyword-only "
                    "arguments to delcare your pass parametes."
                )
        elif param.kind in _unsupportedargs:
            raise TypeError(
                "A pass function must not use variable length arguments, "
                "please use keyword-only arguments"
            )

        # It's keyword-only argument, so record the param name
        # with the default value if it has been given.
        elif (default := param.default) is Parameter.empty:
            params.append(param_name)
        else:
            optional_params[param_name] = default

    if nb_pos_args != 2:
        raise TypeError(
            "A pass function must take exactly "
            f"tow positional arguments ({nb_pos_args} detected)."
        )

    # Determine the runs_on_type based on provided 'on' param.
    runs_on_type = on
    if not isinstance(runs_on_type, type):
        if not isinstance(runs_on_type, tuple):
            runs_on_type = tuple(runs_on_type)
        if len(runs_on_type) == 0:
            raise ValueError('parameter "on" cannot be empty')
        if len(runs_on_type) != 1:
            # Validate the value since Tree and Forest should not be present in
            # passes that run on several nodes. 
            # This is a limitation that is necessary by design
            # since it is used used to differenciate a FOREST from a TREE pass, etc.
            if any((problematic := t) in _on_cls_2_level for t in runs_on_type):
                raise TypeError(
                    f"a pass cannot run both on {problematic} and on other types"
                )
        else:
            # only one value, so flatten it
            (runs_on_type,) = runs_on_type

    # Determine the runs_on_level
    runs_on_level = _on_cls_2_level.get(runs_on_type, Level.NODE)

    proto = PassPrototype(
        do_pass,
        name=do_pass.__name__,
        kind=kind,
        runs_on=runs_on_level,
        runs_on_type=runs_on_type,
        params=params,  # type: ignore[arg-type]
        optional_params=optional_params,
        dependencies=dependencies or (),  # type: ignore[arg-type]
       
        pass_options=options,  # including cached, immutable, etc...
    )

    return proto


def _pass_decorator(**kwargs: Any) -> Callable[[Callable], PassPrototype]:
    """
    Wraps L{new_pass_prototype} to be used as a decorator.
    """

    def decorator(function: Callable) -> PassPrototype:
        return new_pass_prototype(function, **kwargs)

    return decorator


transformation = partial(_pass_decorator, kind=PassKind.TRANSFORMATION)
"""
Main decorators to create a transformation. 

"""

analysis = partial(_pass_decorator, kind=PassKind.ANALYSIS)
"""
Main decorators to create an analysis. 

"""
