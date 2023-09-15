#### Type variable support
# Copyright (c) 2016, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import inspect
import sys

from typing import Iterator, NamedTuple, Sequence, Mapping
from .unionfind import UnionFind
from .typeinfer import Type, TypeVariable
from .._lib.model import Arg, Func, Def
from .._lib.exceptions import (NodeLocation, StaticTypeMismatch, 
                               StaticCodeUnsupported, StaticValueError, StaticException)

class MarkedSubstitutions(NamedTuple):
    uf: UnionFind[Type]
    graph: dict[int, Type]

class Substitutions:
    """
    A domain of instantiated type variables.

    >>> d = Substitutions()
    >>> assert len(list(d.variables())) == 0
    >>> t1 = TypeVariable()
    >>> t2 = TypeVariable()
    >>> v1 = Type('int')
    >>> v2 = Type('float')
    >>> d.bind(t1, v1)
    >>> dd = d.copy()
    >>> d.bind(t2, v2)
    >>> d
    <Substitutions: 
    @TypeVar... -> int
    @TypeVar... -> float
    <UnionFind:
    elts=['@TypeVar...', 'int', '@TypeVar...', 'float'],
    siz=[1, 1, 1, 1],
    par=[0, 1, 2, 3],
    nb_rm=[0, 0, 0, 0],
    removed=[],
    free=[],
    n_elts=4,n_comps=4>
    >
    >>> d.bind(t1, t2)
    >>> d
    <Substitutions: 
    @TypeVar... -> float
    @TypeVar... -> float
    <UnionFind:
    elts=['@TypeVar...', 'int', '@TypeVar...', 'float'],
    siz=[2, 2, 1, 1],
    par=[0, 1, 0, 1],
    nb_rm=[0, 0, 0, 0],
    removed=[],
    free=[],
    n_elts=4,n_comps=2>
    >
    >>> dd.bind(t1, t2)
    >>> dd
    <Substitutions: 
    @TypeVar... -> int
    @TypeVar... -> int
    <UnionFind:
    elts=['@TypeVar...', 'int', '@TypeVar...'],
    siz=[2, 1, 1],
    par=[0, 1, 0],
    nb_rm=[0, 0, 0],
    removed=[],
    free=[],
    n_elts=3,n_comps=2>
    >
    """
    __slots__ = '_uf', '_graph'

    def __init__(self) -> None:
        self._uf: UnionFind[Type] = UnionFind()
        """
        Dans la structure de données Union-find, les classes d'équivalence ne contiennent soit que des variables 
        soit que des termes complexes. Pour passer d'une variable à un terme complexe, 
        on utilise les pointeurs éventuels qui sont dans le graphe. 
        """
        self._graph: dict[int, Type] = {}
    
    def __repr__(self) -> str:
        if len(self._uf):
            r = ['<Substitutions: ']
            r.extend((f'{v.annotation} -> {getattr(self.type(v), "annotation", "<no type>")}' for v in self.variables()))
            r.extend(str(self._uf).splitlines())
            r.append('>')
            return '\n'.join(r)
        else:
            return '<Substitutions>'
    
    def variables(self) -> Iterator[TypeVariable]:
        """
        Iterate over all type variables in the domain.
        """
        return (v for v in self._uf if isinstance(v, TypeVariable))
    
    def _reconcile_types(self, t1:Type, t2:Type) -> Type:
        m = self.mark()
        try:
            nt = reconcile_unify(t1, t2, self)
        except StaticException:
            self.restore(m)
            raise

        self._uf.union(t2, nt)
        self._uf.union(t1, nt)
        return nt

    def bind(self, var:TypeVariable, value:Type) -> None:
        """
        Add a new binding for a variable.
        """
        assert isinstance(var, TypeVariable)
        t = self.type(var)

        if isinstance(value, TypeVariable):
            t2 = self.type(value)
            self._uf.union(var, value)
            rv = self.rep(var)

            # make sure to update de graph with 
            # the most recent representant since it can change after 
            # each call to union() or remove().
            if t2 and t:
                if t2 != t:
                    self._graph[rv] = self._reconcile_types(t2, t)
                else:
                    self._graph[rv] = t
            elif t:
                self._graph[rv] = t
            elif t2:
                self._graph[rv] = t2
            return
        
        if t:
            if t != value:
                self._graph[rv] = self._reconcile_types(t, value)
        else:
            self._uf.add(value)
            self._graph[rv] = value
    
    def rep(self, t:Type) -> int:
        """
        Get the representant of this type.
        """
        self._uf.add(t)
        return self._uf._find(t)

    def type(self, var:TypeVariable) -> Type | None:
        """
        This will return the complex type associated with a given type variable.
        """
        return self._graph.get(self.rep(var))
        

    def mark(self) -> MarkedSubstitutions:
        """
        Create a mark of this substitutions, to be applied with `restore`.
        """
        return MarkedSubstitutions(self._uf.copy(), 
                                   self._graph.copy())

    def restore(self, mark: MarkedSubstitutions) -> None:
        """
        Restore a mark previously created with `mark`.
        """
        self._uf = mark.uf
        self._graph = mark.graph
    
    def copy(self) -> Substitutions:
        """
        Copy the substitutions into a new instance.
        """
        new = Substitutions()
        new.restore(self.mark())
        return new

def substitute(term:Type, subst:Substitutions) -> Type:
    if isinstance(term, TypeVariable):
        replacement = subst.type(term)
        if replacement:
            return replacement
        else:
            return term
    elif term.args:
        new_args = []
        for arg in term.args:
            new_args.append(substitute(arg, subst))
        return term._replace(args=new_args)
    else:
        return term

def occurs_in_type(v:TypeVariable, type2:Type) -> bool:
    """Checks whether a type variable occurs in a type expression.

    Note: Must be called with v pre-pruned

    Args:
        v:  The TypeVariable to be tested for
        type2: The type in which to search

    Returns:
        True if v occurs in type2, otherwise False
    """
    if type2 == v:
        return True
    return occurs_in(v, type2.args)

def occurs_in(t:TypeVariable, types:Sequence[Type]) -> bool:
    """Checks whether a types variable occurs in any other types.

    Args:
        t:  The TypeVariable to be tested for
        types: The sequence of types in which to search

    Returns:
        True if t occurs in any of types, otherwise False
    """
    return any(occurs_in_type(t, t2) for t2 in types)

def reconcile_unify(t1:Type, t2:Type, subst:Substitutions) -> Type:
    """
    When two possible values for a given type variable doesn't unify, 
    we use the first common - nominal - supertype, that's not `object`, or fail. 

    So ``list[str]`` and ``set[str]`` will be reconcile to ``Collection[str]``.
    But ``None`` and ``int`` can't reconcile.
    """
    m = subst.mark()
    try:
        t = unify(t1, t2, subst)
    except StaticException:
        subst.restore(m)
        supertype = t1.supertype
        while 1:
            if supertype is None:
                raise
            if supertype.qualname == 'builtins.object':
                raise
            if supertype.supertype_of(t2):
                break
            supertype = supertype.supertype
        supertype = supertype._replace(args=t1.args)
        t = unify(t2, supertype, subst)
    return t


def approximate_overload(b:Type, types:Sequence[Type], subst:Substitutions) -> Type:
    def try_unify(t:Type, ts:Sequence[Type]) -> Type|None:
        if t.is_typevar:
            return None
        if any(tp.is_typevar for tp in ts):
            return None
        # overapproximate 't'
        new_args = list(t.args)
        for i, tt in enumerate(t.args):
            its = [tp.args[i] for tp in ts]
            if any(it.is_typevar for it in its):
                continue
            it0 = its[0]
            it0ntypes = len(it0.args)
            if all(((it.name == it0.name) and (len(it.args) == it0ntypes)) for it in its):
                ntypes = [TypeVariable() for _ in range(it0ntypes)]
                new_tt = it0._replace(args=ntypes)
                # new_tt.__class__ = it0.__class__
                tt = unify(tt, new_tt, subst)
                new_args[i] = try_unify(tt, [it for it in its]) or tt
        return t._replace(args=new_args)
    
    r = try_unify(b, types)
    if r is None:
        # for the sake of error reporting, we transform 
        # the potential callables into a overload type. 
        raise StaticTypeMismatch(node=b, other=Type.overload.add_args(types), 
                                 desrc='overload approximation failed')
    return r

def better_signature(callable_type:Type) -> inspect.Signature | None:
    """
    Get a signature from the function def information wrapped by this type instance.
    Return None if there is no function wrapped, in the case of a written 'Callable' annotation for instance.
    """
    definition = callable_type.get_meta('definition', Def)
    if not isinstance(definition, Func):
        return None
    parameters: list[inspect.Parameter] = []
    for i, arg in enumerate(callable_type.args[:-1]):
        argdef = arg.get_meta('definition', Arg)
        if argdef is None:
            # the type vars seem to replace the type here:/ with no informations anymore.
            msg = f'missing definition of callable {definition.name()!r} argument {i}: {arg}'
            raise StaticCodeUnsupported(callable_type, msg)
        parameters.append(argdef.to_parameter().replace(annotation=arg))
    return inspect.Signature(parameters, return_annotation=callable_type.args[-1])

def callable_struct(callable_type:Type) -> tuple[Sequence[Type], Mapping[str, Type], Type]:
    """
    Returns tuple: (args, kwargs, rtype)
    """
    if not callable_type.args:
        return [], {}, Type.Any

    positionals: list[Type] = []
    keywords: dict[str, Type] = {}

    for a in callable_type.args[:-1]:
        keyword = a.get_meta('keyword', str)
        if keyword is not None:
            keywords[keyword] = keywords.setdefault(keyword, Type.Any).merge(a)
        else:
            if keywords:
                raise StaticValueError(callable_type, 
                    'invalid callable, keywords came before positionals :/')
            positionals.append(a)
    
    return positionals, keywords, callable_type.args[-1]

def poor_signature(callable_type:Type) -> inspect.Signature:
    """
    Get a approximated signature based on a Callable type 
    (created from annotations or from ast.Call before unification). 
    """
    positionals, keywords, rtype = callable_struct(callable_type)
    # simplify signatures
    parameters: list[inspect.Parameter] = []
    parameters.extend((inspect.Parameter(f'__{i}__', 
                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       annotation=t) for i,t in enumerate(positionals)))
    parameters.extend((inspect.Parameter(name,
                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       annotation=t) for name,t in keywords.items()))
    return inspect.Signature(parameters, return_annotation=rtype)

def bind_callables(a:Type, b:Type) -> tuple[Type, Type, inspect.BoundArguments]:
    """
    Dertemine signature based compatibility between two callables.
    Returns (caller, callee, bound arguments).
    """
    def _add_err_note(e:Exception):
        if sys.version_info >= (3, 11):
            shortsig = inspect.Signature(
                [p.replace(annotation=p.annotation.annotation,
                            default=inspect.Parameter.empty,
                            ) for p in sig.parameters.values()])
            
            e.add_note(f'signature={str(shortsig)}')
            e.add_note(f'args={[a.annotation for a in args]}')
            e.add_note(f'keywords={dict((n,a.annotation) for n,a in keywords.items())}')

    caller, callee = a, b
    sig = better_signature(callee) or poor_signature(callee)
    args, keywords, _ = callable_struct(caller)
    assert len(args)+len(keywords)+1==len(caller.args)
    try:
        bargs = sig.bind(*args, **keywords)
    except Exception as e:
        _add_err_note(e)

        caller, callee = b, a
        sig = better_signature(callee) or poor_signature(callee)
        args, keywords, _ = callable_struct(caller)
        assert len(args)+len(keywords)+1==len(caller.args)

        try:
            bargs = sig.bind(*args, **keywords)
        except Exception as e:
            _add_err_note(e)
            raise StaticTypeMismatch(a, other=b) from e
    
    return caller, callee, bargs

def unify(t1:Type, t2:Type, subst:Substitutions|None=None) -> Type:
    """Unify the two types t1 and t2.

    Returns a new type that represent the unification of both type.

    Args:
        t1: The first type to be made equivalent
        t2: The second type to be be equivalent
        subst: type variables substitutions.

    Returns:
        Tuple: (The unified type, Subst)

    Raises:
        InferenceError: Raised if the types cannot be unified.
    
    >>> t = Type.overload.add_args(args=
    ... [Type.Callable.add_args([Type('float', 'builtins'), Type('int', 'builtins'),]), 
    ...  Type.Callable.add_args([Type.Union.add_args([Type('str', 'builtins'), 
    ...                                               Type('bytes', 'builtins'),
    ...                                               Type('object', 'builtins')]), 
    ...                          Type('str', 'builtins'),])])
    >>> expr1 = Type.Callable.add_args([Type('int', 'builtins'), TypeVariable()])
    >>> print(t.annotation)
    (float) -> int | (str | bytes | object) -> str
    >>> print(expr1.annotation)
    (int) -> @TypeVar...
    >>> print(unify(expr1, t).annotation)
    (float) -> int
    >>> expr2 = Type.Callable.add_args([Type('float', 'builtins'), TypeVariable()])
    >>> print(unify(expr2, t).annotation)
    (float) -> int
    >>> expr3 = Type.Callable.add_args([Type('bytes', 'builtins'), TypeVariable()])
    >>> print(unify(expr3, t).annotation)
    (object) -> str
    >>> t2 = Type.Callable.add_args([Type('list', 'builtins').add_args(
    ...     [Type('@TypeVar1')]), Type('list', 'builtins').add_args(
    ...     [Type('@TypeVar1')]),])
    >>> expr4 = Type.Callable.add_args([Type('list', 'builtins').add_args([Type('float', 'builtins')]), Type('@TypeVar2')])
    >>> s = Substitutions()
    >>> print(unify(expr4, t2, s).annotation)
    (list[float]) -> list[float]
    """
    subst = Substitutions() if subst is None else subst

    if t1 == t2:
        return substitute(t1, subst)
    elif isinstance(t2, TypeVariable) and not isinstance(t1, TypeVariable):
        return unify(t2, t1, subst)
    elif isinstance(t1, TypeVariable):
        if occurs_in_type(t1, t2):
            raise StaticCodeUnsupported(t1, "recursive unification")
        subst.bind(t1, t2)
        return substitute(t2, subst)
    elif t2.is_overload and not t1.is_overload:
        return substitute(unify(t2, t1, subst), subst)
    elif t1.is_overload:
        types: list[tuple[Type, MarkedSubstitutions]] = []
        errs: list[Exception] = []
        
        m = subst.mark()
        for t in t1.args:
            try:
                types.append((unify(t, t2, subst), subst.mark()))
            except StaticException as e:
                errs.append(e)
            # do not pass substitutions in between overloads
            subst.restore(m)

        if types:
            subst.restore(types[0][1])

            if len(types) == 1:
                return substitute(unify(types[0][0], t2, subst), subst)
            # too many overloads are found, so extract as many 
            # information as we can, and unify the rest with the first overload match.
            return substitute(
                        unify(types[0][0], 
                              approximate_overload(t2, [_t[0] for _t in types],
                                                    subst), 
                              subst), 
                        subst)
        else:
            raise StaticTypeMismatch(t1, other=t2, 
                    desrc='none of the overloads matches')
    elif t2.is_union and not t1.is_union:
        return substitute(unify(t2, t1, subst), subst)
    elif t1.is_union:
        utypes: list[Type] = []
        errs = []
        for t in t1.args:
            m = subst.mark()
            try:
                utypes.append(substitute(unify(t, t2, subst), subst))
            except StaticException as e:
                errs.append(e)
                subst.restore(m)
            # for unions, we do pass substitutions in between elements
        if utypes:
            new_type = Type.Any
            for t in utypes:
                new_type = new_type.merge(t)

            return new_type
        else:
            raise StaticTypeMismatch(t1, other=t2, desrc="none of the union element matches")
    elif t2.unknown:
        return substitute(t1, subst)
    elif t1.unknown:
        return substitute(t2, subst)
    elif t2.is_callable and t1.is_callable:
        caller, callee, bargs = bind_callables(t1, t2)
        # now let's adjust callee's arguments to match caller's
        # transfer typevars of course!
        new_args = [*(bargs.signature.parameters[a].annotation 
                      for a in bargs.arguments), callee.args[-1]]

        if new_args != list(callee.args):
            callee = callee._replace(args=new_args)

        assert len(callee.args) == len(caller.args), f'{callee.args} != {caller.args}'
        
        # Unify arguments
        unified_args: list[Type] = []
        for p, q in zip(callee.args, caller.args):
            t = substitute(unify(p, q, subst), subst)
            unified_args.append(t)
        
        return callee._replace(args=unified_args)
    
    elif len(t1.args) != len(t2.args):
        raise StaticTypeMismatch(t1, other=t2, desrc='arity differs')
   
    # catch-all case.
    if t2.qualname == t1.qualname or t1.supertype_of(t2):
        keep = t1
    elif t2.supertype_of(t1):
        keep = t2
    else:
        raise StaticTypeMismatch(t1, other=t2, desrc='incompatible types')
    
    # Unify arguments
    unified_args = []
    for p, q in zip(t1.args, t2.args):
        t = substitute(unify(p, q, subst), subst)
        unified_args.append(t)
    
    return keep._replace(args=unified_args)

if __debug__:
    unify_org = unify
    def debug_unify(t1:Type, t2:Type, subst:Substitutions|None=None) -> Type:
        subst = Substitutions() if subst is None else subst

        subst_before = subst.copy()
        try:
            r = unify_org(t1, t2, subst)
        except StaticException:
            print(f'Unification of:\n- {t1.annotation}\n- {t2.annotation}\n=> MISMATCH\n Typevars before: {repr(subst_before)})')
            raise
        else:
            subst_after = subst
            print(f'Unification of:\n- {t1.annotation}\n- {t2.annotation}\n=> {r.annotation}')
            # simply print the diff the of repr
            import difflib
            print('\n'.join(difflib.context_diff(repr(subst_before).splitlines(), 
                                                 repr(subst_after).splitlines(),
                                                 fromfile='before', 
                                                 tofile='after', )))
            return r
        
    # unify = debug_unify