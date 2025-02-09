"""
Generic data structures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
import operator
import types
from typing import (
    Callable,
    Collection,
    Generic,
    Hashable,
    Iterator,
    Any,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    overload,
    TYPE_CHECKING,
)

from beniget.ordered_set import ordered_set as _oset # type: ignore

if TYPE_CHECKING:
    from typing import TypeAlias
    OrderedSet: TypeAlias = set
else:
    OrderedSet = _oset

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

################# Generic immutable-like lazy sequence

class LazySeq(Sequence[_T]):
    """
    A lazy sequence makes an iterator looks like an immutable sequence.
    """

    __slots__ = '_iterator', '_values'

    def __init__(self, iterable:Iterable[_T]) -> None:
        self._iterator = iter(iterable)
        self._values: List[_T] = []
    
    def _curr(self,) -> int:
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

################# Generic immutable-like lazy mapping

class LazyMap(Mapping[_KT, _VT]):
    """
    A lazy map makes an iterator look like an immutable mapping.
    """

    __slots__ = '_dict', '_iterator'

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

################# Generic immutable ChainMap

class ChainMap(Mapping['_KT', '_VT']):
    """
    Combine multiple mappings for sequential lookup.

    For example, to emulate Python's normal lookup sequence:

        import __builtin__
        pylookup = ChainMap((locals(), globals(), vars(__builtin__)))        
    """

    __slots__ = '_maps',

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

################# Generic immuatble mapping

# TODO: Rename me FrozenMap, a dict is always mutable.
class FrozenDict(Mapping['_KT', '_VT']):
    """
    An immutable mapping.

    Example usage:

    >>> fd = FrozenDict(a=1, b=2)
    >>> fd['a']
    1
    >>> fd['b']
    2
    >>> list(fd)
    ['a', 'b']
    >>> len(fd)
    2
    >>> repr(fd)
    "{'a': 1, 'b': 2}"
    >>> str(fd)
    "{'a': 1, 'b': 2}"
    >>> hash(fd) == hash(FrozenDict(a=1, b=2))
    True
    >>> hash(fd) != hash(FrozenDict(a=2, b=1))
    True
    >>> fd['c']
    Traceback (most recent call last):
        ...
    KeyError: 'c'
    """

    __slots__ = '_d', '_hash'
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

################# Generic namespace

# TODO: Can this be optimized to use __slots__? 
class FrozenNamespace(types.SimpleNamespace):
    """
    An immutable namespace.

    Example usage:

    >>> fn = FrozenNamespace(a=1, b=2)
    >>> fn.a
    1
    >>> fn.b
    2
    >>> fn.c
    Traceback (most recent call last):
        ...
    AttributeError: 'FrozenNamespace' object has no attribute 'c'
    >>> fn.a = 3
    Traceback (most recent call last):
        ...
    AttributeError: 'FrozenNamespace' object is read-only
    >>> print(repr(fn))
    FrozenNamespace(a=1, b=2)
    >>> hash(fn) == hash(FrozenNamespace(a=1, b=2))
    True
    >>> hash(fn) != hash(FrozenNamespace(a=2, b=1))
    True
    >>> fn == FrozenNamespace(a=1, b=2)
    True
    >>> fn == FrozenNamespace(a=2, b=1)
    False
    >>> fn.__dict__
    {'a': 1, 'b': 2, '_FrozenNamespace__hash': ...}
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__freeze__()
    
    def __freeze__(self):
        self.__hash = hash(FrozenDict(self.__dict__.items()))+1
    
    def __is_frozen__(self):
        return f"_{type(self).__name__}__hash" in self.__dict__
    
    def __hash__(self):
        if self.__is_frozen__():
            return self.__hash
        raise TypeError(f"{type(self).__name__!r} object is not hashable yet")
    
    def __repr__(self):
        # repr() only consider public attributes because of ugly '_FrozenNamespace__hash' attribute.
        content = ", ".join(f"{k}={v!r}" for k,v in self.__dict__.items() if not k.startswith('_'))
        return f"{type(self).__name__}({content})"

    def __setattr__(self, name, value):
        if self.__is_frozen__():
            raise AttributeError(f"{type(self).__name__!r} object is read-only")
        else:
            super().__setattr__(name, value)


################# Generic Result object (inspired by Rust)

class CallResult(Generic[_T]):
    # TODO: Add proper typing
    """
    Simple wrapper for the result of a function call. 

    The result can either be a success, in which case the C{result} attribute will give the return value of the function.
    Or the result can be an error, in which case the C{error} property returns an expection instance accessing the C{result} attribute
    will raise the exception.
    """
    
    @property
    def result(self) -> _T:
        raise NotImplementedError(self.result)
    
    @property
    def error(self) -> Exception | None:
        if isinstance(self, _Error):
            return self._error
        return None

    @classmethod
    def new(cls, obj: _T | Exception) -> CallResult:
        if isinstance(obj, Exception):
            return _Error(obj)
        return _Success(obj)

@dataclass(frozen=True)
class _Error(CallResult):
    _error: Exception

    @property
    def result(self) -> object:
        raise self._error

@dataclass(frozen=True)
class _Success(CallResult):
    _result: object

    @property
    def result(self) -> object:
        return self._result

################# Generic Indexer implementation

_SeriesOfKeysT = TypeVar("_SeriesOfKeysT", bound=Sequence[Hashable])

class Indexer(Generic[_SeriesOfKeysT]):
    """
    Indexer for a series of keys.

    The index helps with searching which series of keys contains a specific key value.

    Example with three keys:

    >>> index = Indexer(keys=["key1", "key2", "key3"], skipKeys=["key3"])

    Adding entries (key3 won't be indexed):

    >>> index.add(("a", "b", "c"))
    >>> index.add(("a", "x", "y"))
    >>> index.add(("m", "n", "z"))

    Searching for entries by key1:

    >>> sorted(index.search(key1="a"))  # Find all entries with key1 = "a"
    [('a', 'b', 'c'), ('a', 'x', 'y')]
    >>> list(index.search(key1="m"))  # Find all entries with key1 = "m"
    [('m', 'n', 'z')]

    Searching with multiple keys:

    >>> list(index.search(key1="a", key2="b"))  # Find entry with key1 = "a" and key2 = "b"
    [('a', 'b', 'c')]
    >>> list(index.search(key1="a", key2="z"))  # No entries with key2 = "z"
    []

    Discarding a non-existent entry:

    >>> index.discard(("a", "b", "z"))  # Discarding non-existent key, no error
    >>> sorted(index.search(key1="a"))  # Confirm original entry is still present
    [('a', 'b', 'c'), ('a', 'x', 'y')]
    >>> sorted(index.search(key1="m"))  # Confirm other entry is still present
    [('m', 'n', 'z')]

    Discarding an entry:

    >>> index.discard(("a", "b", "c"))  # Remove the ("a", "b", "c") entry
    >>> list(index.search(key1="a"))  # Now only ("a", "x", "y") should remain
    [('a', 'x', 'y')]

    Retrieving distinct values for a key:

    >>> sorted(index.kvalues("key1"))  # List all distinct values for key1
    ['a', 'm']
    >>> sorted(index.kvalues("key2"))  # List all distinct values for key2
    ['n', 'x']
    >>> index.add(("1", "b", "4"))
    >>> sorted(index.kvalues("key2"))  # List all distinct values for key2
    ['b', 'n', 'x']

    Attempting to retrieve values for a skipped key:

    >>> index.kvalues("key3")  # key3 is skipped, so it should raise an error
    Traceback (most recent call last):
    ...
    TypeError: Unexpected key: key3

    Invalid searches and errors:

    >>> index.search()  # No keys provided for search
    Traceback (most recent call last):
    ...
    TypeError: Expected at least one keyword argument
    >>> index.search(nonexistent="z")  # Search with an invalid key
    Traceback (most recent call last):
    ...
    TypeError: Unexpected keyword: nonexistent
    """
    def __init__(self, keys: Collection[str], skipKeys: Collection[str]) -> None:
        """
        @param keys: An ordered collection of the names of the chache keys.
        @param skipKeys: A collection of key names that should not be indexed.
        """
        self.__keys = keys
        self.__skipKeys = skipKeys

        self.__store: dict[str, dict[Hashable, set[_SeriesOfKeysT]]] = (
            defaultdict(lambda: defaultdict(OrderedSet))   )
    
    def add(self, key: _SeriesOfKeysT) -> None:
        # O(1)
        for label, value in zip(self.__keys, key): # TODO: use strict=True
            if label in self.__skipKeys:
                continue
            self.__store[label][value].add(key)
    
    def discard(self, key: _SeriesOfKeysT) -> None:
        # O(1)
        for label, value in zip(self.__keys, key): # TODO: use strict=True
            if label in self.__skipKeys:
                continue
            self.__store[label][value].discard(key)
            # TODO: Is it worth it to delete empty sets from the structure ?
            # This would optimize kvalues() so tat we don't have to return a new set
            # but kvalues is only used in the tests at this time...

    def search(self, **key: Hashable) -> Collection[_SeriesOfKeysT]: # typed as Collection so it cannot be mutated.
            # O(min(len(s) for s in set of keys)) or O(1) if only one key is provided
            # Verify no junk parmeters.
            if not key:
                raise TypeError(f'Excepted at least one keyword argument')
            if not all(invalid:=(k in self.__keys) and (invalid:=k not in self.__skipKeys) for k in key):
                raise TypeError(f'Unexpected keyword: {invalid}')
            
            sets = [self.__store[label][value] for label, value in key.items()]
            # Fast track if only one key is provided
            if len(sets) == 1:
                return sets[0]
            # Create the intersection of sets starting with the smallest for performance reasons.
            sets.sort(key=len)
            return reduce(operator.and_, sets)

    def kvalues(self, key: str) -> Collection[Hashable]:
        # Verify no junk parmeters.
        if key not in self.__keys or key in self.__skipKeys:
            raise TypeError(f'Unexpected key: {key}')
        # Ignore empty sets.
        return OrderedSet(k for k,v in self.__store[key].items() if v)

################# Generic Cache implementation

class Cache(Generic[_SeriesOfKeysT, _VT]):
    """
    Generic cache that stores a mapping of keys to values and supports searching and retrieval operations.

    Example with a cache of 3-part keys and skipping the third key:

    >>> cache: Cache[tuple[str, str, str], str] = Cache(keys=["key1", "key2", "key3"], skipKeys=["key3"])

    Adding entries to the cache:

    >>> cache.set(("a", "b", "c"), "value1")
    >>> cache.set(("x", "y", "z"), "value2")
    >>> cache.set(("m", "n", "o"), "value3")

    Retrieving entries from the cache:

    >>> cache.get(("a", "b", "c"))  # Retrieve the value for the key ("a", "b", "c")
    'value1'
    >>> cache.get(("x", "y", "z"))  # Retrieve the value for the key ("x", "y", "z")
    'value2'
    >>> print(cache.get(("nonexistent", "key", "set")))  # Try to retrieve a non-existent key
    None

    Discarding entries from the cache:

    >>> cache.remove(("a", "b", "c"))  # Remove the entry with key ("a", "b", "c")
    >>> print(cache.get(("a", "b", "c")))  # Check that the key is no longer present
    None
    >>> list(cache.search(key1="a"))  # Confirm that the key has been removed from the index
    []
    """

    def __init__(self, keys: Collection[str], skipKeys: Collection[str]) -> None:
        """
        @param keys: An ordered collection of the names of the chache keys.
        @param skipKeys: A collection of key names that should not be indexed.
        """
        self.__store: dict[_SeriesOfKeysT, _VT] = {}
        self.__indexer: Indexer[_SeriesOfKeysT] = Indexer(keys, skipKeys)
    
    def set(self, key:_SeriesOfKeysT, value:_VT) -> None:
        """
        Adds a new result to the cache. 
        This overrides any entry already present in the cache with the same key.
        """
        # O(1)
        self.__store[key] = value
        self.__indexer.add(key)
    
    def remove(self, key:_SeriesOfKeysT) -> None:
        """
        Removes a key from the cache and its associated result.
        The key must be present in the cache, otherwise it fails with ValueError.
        """
        # O(1)
        if key not in self.__store:
            raise ValueError('key not in cache')
        del self.__store[key]
        self.__indexer.discard(key)

    def get(self, key:_SeriesOfKeysT) -> _VT | None:
        """
        Retrieves the result associated with the given key from the cache.
        """
        # O(1)
        return self.__store.get(key)
    
    def search(self, **key: Hashable) -> Collection[_SeriesOfKeysT]:
        """
        Searches the cache for entries matching the given partial key(s).
        """
        return self.__indexer.search(**key)
    
    def kvalues(self, key: str) -> Collection[Hashable]:
        """
        Returns all distinct values for a particular key in the cache.
        """
        return self.__indexer.kvalues(key)
    
    def allkeys(self) -> Collection[_SeriesOfKeysT]:
        return list(self.__store)

################# Generic callable proxy

class GetProxy(Generic[_T, _VT]):
    """
    Provide L{get} and L{__cal__} methods that defers to an underlying callable 
    taking any number of positional or keywords arguments . 
    """
    def __init__(self, factory: Callable[..., _VT]):
        self._factory = factory

    def get(self, 
            *element: _T,
            **kwargs: Hashable) -> _VT:
        """
        Request a value from this proxy. 
        """
        return self._factory(*element, **kwargs)

    __call__ = get
