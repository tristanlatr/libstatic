"""
Generic data structures.
"""

from __future__ import annotations

from typing import (
    Iterator,
    Any,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

_T = TypeVar("_T")
class LazySeq(Sequence[_T]):
    """
    A lazy sequence makes an iterator look like an immutable sequence.
    """
    def __init__(self, iterable:Iterable[_T]) -> None:
        self._iterator = iter(iterable)
        self._values: List[_T] = []
    
    def _curr(self,) ->int:
        return len(self._values)-1
    
    def _consume_next(self) -> _T:
        val = next(self._iterator)
        self._values.append(val)
        return val
    
    def _consume_until(self, key:int) -> None:
        if key < 0:
            self._consume_all()
            return
        while self._curr() < key:
            try:
                self._consume_next()
            except StopIteration:
                break
    
    def _consume_all(self) -> None:
        while 1:
            try:
                self._consume_next()
            except StopIteration:
                break
    @overload
    def __getitem__(self, key:int) -> _T:
        ...
    @overload
    def __getitem__(self, key:slice) -> list[_T]:
        ...
    def __getitem__(self, key:int|slice) -> _T | list[_T]:
        if isinstance(key, int):
            self._consume_until(key)
        else:
            self._consume_all()
        return self._values[key]
    
    def __iter__(self) -> Iterator[_T]:
        yield from self._values
        while 1:
            try:
                yield self._consume_next()
            except StopIteration:
                break
    
    def __len__(self) -> int:
        self._consume_all()
        return len(self._values)

    def __bool__(self) -> bool:
        if self._curr() > -1:
            return True
        try:
            self._consume_next()
        except StopIteration:
            return False
        return True

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

class LazyMap(Mapping[_KT, _VT]):
    """
    A lazy map makes an iterator look like an immutable mapping.
    """
    def __init__(self, iterator:Iterator[Tuple[_KT, _VT]]):
        self._dict: dict[_KT, _VT] = {}
        self._iterator = iterator
    
    def _curr(self,) ->int:
        return len(self._dict)-1
    
    def _consume_next(self) -> Tuple[_KT, _VT]:
        k,v = next(self._iterator)
        self._dict[k] = v
        return k,v
    
    def _consume_all(self) -> None:
        while 1:
            try:
                self._consume_next()
            except StopIteration:
                break

    def __getitem__(self, key:_KT) -> _VT:
        if key in self:
            return self._dict[key]
        else:
            raise KeyError(key)

    def __contains__(self, key:object) -> bool:
        if key in self._dict:
            return True
        while 1:
            try:
                k, _ = self._consume_next()
            except StopIteration:
                return False
            if k is key:
                return True
    
    def __iter__(self) -> Iterator[_KT]:
        yield from self._dict
        while 1:
            try:
                k, _ = self._consume_next()
            except StopIteration:
                break
            yield k
    
    def __len__(self) -> int:
        self._consume_all()
        return len(self._dict)

class ChainMap(Mapping['_KT', '_VT']):
    """
    Combine multiple mappings for sequential lookup.

    For example, to emulate Python's normal lookup sequence:

        import __builtin__
        pylookup = ChainMap((locals(), globals(), vars(__builtin__)))        
    """

    def __init__(self, maps:Sequence[Mapping[_KT, _VT]]) -> None:
        self._maps = maps

    def __getitem__(self, key:_KT) ->_VT:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __len__(self) -> int:
        return len(set().union(*self._maps))     # reuses stored hash values if possible

    def __iter__(self) -> Iterator[_KT]:
        d = {}
        for mapping in reversed(self._maps):
            d.update(dict.fromkeys(mapping))    # reuses stored hash values if possible
        return iter(d)

class FrozenDict(Mapping['_KT', '_VT']):
    # copied from https://stackoverflow.com/a/2704866

    def __init__(self, *args:Any, **kwargs:Any):
        self._d = dict(*args, **kwargs)
        self._hash:int|None = None

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, key:_KT) -> _VT:
        return self._d[key]
    
    def __repr__(self) -> str:
        return repr(self._d)
    
    def __str__(self) -> str:
        return str(self._d)

    def __hash__(self) -> int:
        # It would have been simpler and maybe more obvious to 
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of 
        # n we are going to run into, but sometimes it's hard to resist the 
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash
